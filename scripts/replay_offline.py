#!/usr/bin/env python3
"""
Offline replay coach that processes GSI logs and generates coaching suggestions.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Iterator, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gsi.schema import GSISnapshot, parse_gsi_payload
from src.coach.rules import ComboCoach
from src.coach.rl_bridge import CombinedRLCoach

logger = logging.getLogger(__name__)


class GSILogReplayer:
    """Replay GSI logs for offline coaching analysis."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.total_snapshots = 0
        self.valid_snapshots = 0
        
    def count_snapshots(self) -> int:
        """Count total snapshots in log file."""
        if self.total_snapshots > 0:
            return self.total_snapshots
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        self.total_snapshots += 1
        except Exception as e:
            logger.error(f"Error counting snapshots: {e}")
        
        return self.total_snapshots
    
    def replay(self, start_time: float = 0, end_time: float = float('inf')) -> Iterator[GSISnapshot]:
        """
        Replay GSI snapshots from log file.
        
        Args:
            start_time: Start replay from this game time (seconds)
            end_time: Stop replay at this game time (seconds)
            
        Yields:
            GSISnapshot objects
        """
        try:
            with open(self.log_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse JSON snapshot
                        data = json.loads(line)
                        snapshot = GSISnapshot.from_gsi_payload(data)
                        
                        # Filter by time range
                        if snapshot.map_state:
                            game_time = snapshot.map_state.game_time
                            if game_time < start_time or game_time > end_time:
                                continue
                        
                        self.valid_snapshots += 1
                        yield snapshot
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Error parsing snapshot on line {line_num}: {e}")
                        
        except FileNotFoundError:
            logger.error(f"Log file not found: {self.log_file}")
        except Exception as e:
            logger.error(f"Error reading log file: {e}")


class CoachingMetrics:
    """Track coaching performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_suggestions = 0
        self.suggestions_by_type = {}
        self.suggestions_by_priority = {1: 0, 2: 0, 3: 0}
        
        # Last-hit specific metrics
        self.initial_cs = 0
        self.final_cs = 0
        self.initial_denies = 0
        self.final_denies = 0
        self.initial_gold = 0
        self.final_gold = 0
        self.initial_time = 0
        self.final_time = 0
        
        # Timing tracking
        self.cs_timeline = []  # (time, cs_count) pairs
        self.gold_timeline = []  # (time, gold) pairs
        
    def update_snapshot_metrics(self, snapshot: GSISnapshot):
        """Update metrics from snapshot."""
        if not snapshot.player or not snapshot.map_state:
            return
        
        game_time = snapshot.map_state.game_time
        cs_count = snapshot.player.last_hits
        deny_count = snapshot.player.denies
        gold = snapshot.player.gold
        
        # Initialize if first snapshot
        if self.initial_time == 0:
            self.initial_cs = cs_count
            self.initial_denies = deny_count
            self.initial_gold = gold
            self.initial_time = game_time
        
        # Update final values
        self.final_cs = cs_count
        self.final_denies = deny_count
        self.final_gold = gold
        self.final_time = game_time
        
        # Track timeline
        self.cs_timeline.append((game_time, cs_count))
        self.gold_timeline.append((game_time, gold))
    
    def record_suggestions(self, suggestions: List):
        """Record coaching suggestions."""
        self.total_suggestions += len(suggestions)
        
        for suggestion in suggestions:
            # Count by type
            suggestion_type = suggestion.type
            self.suggestions_by_type[suggestion_type] = self.suggestions_by_type.get(suggestion_type, 0) + 1
            
            # Count by priority
            self.suggestions_by_priority[suggestion.priority] += 1
    
    def calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final performance metrics."""
        duration_minutes = (self.final_time - self.initial_time) / 60.0
        cs_gained = self.final_cs - self.initial_cs
        denies_gained = self.final_denies - self.initial_denies
        gold_gained = self.final_gold - self.initial_gold
        
        metrics = {
            # Basic game metrics
            "duration_minutes": duration_minutes,
            "cs_total": self.final_cs,
            "cs_gained": cs_gained,
            "cs_per_minute": cs_gained / max(duration_minutes, 1),
            "denies_total": self.final_denies,
            "denies_gained": denies_gained,
            "deny_rate": denies_gained / max(cs_gained, 1),
            "gold_total": self.final_gold,
            "gold_gained": gold_gained,
            "gpm": gold_gained / max(duration_minutes, 1),
            
            # Coaching metrics
            "total_suggestions": self.total_suggestions,
            "suggestions_by_type": dict(self.suggestions_by_type),
            "suggestions_by_priority": dict(self.suggestions_by_priority),
            "suggestions_per_minute": self.total_suggestions / max(duration_minutes, 1),
        }
        
        # CS benchmarks
        if duration_minutes >= 5:
            cs_5min = self._interpolate_cs_at_time(5 * 60)
            metrics["cs_at_5min"] = cs_5min
            metrics["cs_5min_benchmark"] = 38  # Good benchmark
            metrics["cs_5min_ratio"] = cs_5min / 38
        
        if duration_minutes >= 10:
            cs_10min = self._interpolate_cs_at_time(10 * 60)
            metrics["cs_at_10min"] = cs_10min
            metrics["cs_10min_benchmark"] = 80
            metrics["cs_10min_ratio"] = cs_10min / 80
        
        return metrics
    
    def _interpolate_cs_at_time(self, target_time: float) -> float:
        """Interpolate CS count at specific time."""
        if not self.cs_timeline:
            return 0.0
        
        # Find surrounding points
        before = None
        after = None
        
        for time_val, cs_val in self.cs_timeline:
            if time_val <= target_time:
                before = (time_val, cs_val)
            elif time_val > target_time and after is None:
                after = (time_val, cs_val)
                break
        
        if before is None:
            return self.cs_timeline[0][1] if self.cs_timeline else 0.0
        
        if after is None:
            return before[1]
        
        # Linear interpolation
        time_before, cs_before = before
        time_after, cs_after = after
        
        if time_after == time_before:
            return cs_before
        
        ratio = (target_time - time_before) / (time_after - time_before)
        return cs_before + ratio * (cs_after - cs_before)


def analyze_replay(log_file: Path, coach_type: str = "rules", policy_path: str = None,
                  start_time: float = 0, end_time: float = float('inf'),
                  verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze replay with coaching system.
    
    Args:
        log_file: Path to GSI log file
        coach_type: "rules" or "rl" or "combined"
        policy_path: Path to RL policy (for RL coaches)
        start_time: Start analysis from this game time
        end_time: End analysis at this game time
        verbose: Print detailed suggestions
        
    Returns:
        Analysis metrics dictionary
    """
    # Create coach
    if coach_type == "rules":
        coach = ComboCoach()
    elif coach_type == "rl":
        coach = CombinedRLCoach(policy_path)
    elif coach_type == "combined":
        # TODO: Combine rules + RL coaches
        coach = CombinedRLCoach(policy_path)
    else:
        raise ValueError(f"Unknown coach type: {coach_type}")
    
    # Create replayer and metrics tracker
    replayer = GSILogReplayer(log_file)
    metrics = CoachingMetrics()
    
    logger.info(f"Starting replay analysis of {log_file}")
    logger.info(f"Coach type: {coach_type}")
    
    if verbose:
        print(f"\n=== COACHING REPLAY: {log_file.name} ===")
        print(f"Coach: {coach_type}")
        if policy_path:
            print(f"RL Policy: {policy_path}")
        print()
    
    processed_count = 0
    last_progress_time = time.time()
    
    try:
        for snapshot in replayer.replay(start_time, end_time):
            processed_count += 1
            
            # Update metrics from snapshot
            metrics.update_snapshot_metrics(snapshot)
            
            # Get coaching suggestions
            suggestions = coach.analyze(snapshot)
            metrics.record_suggestions(suggestions)
            
            # Print suggestions if verbose
            if verbose and suggestions:
                game_time = snapshot.map_state.game_time if snapshot.map_state else 0
                minutes = int(game_time // 60)
                seconds = int(game_time % 60)
                
                for suggestion in suggestions:
                    priority_str = "!!!" if suggestion.priority == 3 else "!!" if suggestion.priority == 2 else "!"
                    print(f"[{minutes:02d}:{seconds:02d}] {priority_str} {suggestion.type.upper()}: {suggestion.message}")
            
            # Progress reporting
            current_time = time.time()
            if current_time - last_progress_time > 5.0:  # Every 5 seconds
                logger.info(f"Processed {processed_count} snapshots...")
                last_progress_time = current_time
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
    
    # Calculate final metrics
    final_metrics = metrics.calculate_final_metrics()
    final_metrics["snapshots_processed"] = processed_count
    final_metrics["snapshots_total"] = replayer.total_snapshots
    final_metrics["log_file"] = str(log_file)
    final_metrics["coach_type"] = coach_type
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Offline GSI replay coach")
    parser.add_argument("log_file", type=Path, help="GSI log file to analyze")
    parser.add_argument("--coach", choices=["rules", "rl", "combined"], default="rules",
                       help="Coaching system to use")
    parser.add_argument("--policy", type=str, help="Path to RL policy file")
    parser.add_argument("--start-time", type=float, default=0, help="Start time (game seconds)")
    parser.add_argument("--end-time", type=float, default=float('inf'), help="End time (game seconds)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print suggestions during replay")
    parser.add_argument("--output", type=Path, help="Save metrics to JSON file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not args.log_file.exists():
        logger.error(f"Log file not found: {args.log_file}")
        return 1
    
    # Run analysis
    try:
        metrics = analyze_replay(
            log_file=args.log_file,
            coach_type=args.coach,
            policy_path=args.policy,
            start_time=args.start_time,
            end_time=args.end_time,
            verbose=args.verbose
        )
        
        # Print summary
        print(f"\n=== ANALYSIS SUMMARY ===")
        print(f"Duration: {metrics['duration_minutes']:.1f} minutes")
        print(f"CS: {metrics['cs_total']} total, {metrics['cs_per_minute']:.1f}/min")
        print(f"Denies: {metrics['denies_total']} total, {metrics['deny_rate']:.1%} rate")
        print(f"Gold: {metrics['gold_total']} total, {metrics['gpm']:.0f} GPM")
        print(f"Suggestions: {metrics['total_suggestions']} total, {metrics['suggestions_per_minute']:.1f}/min")
        
        # CS benchmarks
        if 'cs_at_5min' in metrics:
            print(f"CS@5min: {metrics['cs_at_5min']:.0f}/{metrics['cs_5min_benchmark']} ({metrics['cs_5min_ratio']:.1%})")
        if 'cs_at_10min' in metrics:
            print(f"CS@10min: {metrics['cs_at_10min']:.0f}/{metrics['cs_10min_benchmark']} ({metrics['cs_10min_ratio']:.1%})")
        
        print(f"\nSuggestion breakdown:")
        for stype, count in metrics['suggestions_by_type'].items():
            print(f"  {stype}: {count}")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())