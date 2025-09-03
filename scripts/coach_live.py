#!/usr/bin/env python3
"""
Live coaching system that tails GSI session files and speaks prompts.

Monitors the newest GSI session file for updates and provides real-time voice coaching.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gsi.schema import GSISnapshot, parse_gsi_payload
from src.coach.rules import get_coaching_prompts
from src.coach.prompt_bus import PromptBus
from src.coach.voice import speak


class LiveCoach:
    """Real-time coaching system that monitors GSI logs and speaks prompts."""
    
    def __init__(self, gsi_dir: str, policy_path: Optional[str] = None, 
                 tau: float = 0.7, voice: Optional[str] = None):
        """
        Initialize live coaching system.
        
        Args:
            gsi_dir: Directory containing GSI session logs
            policy_path: Path to PyTorch policy file (.pt) 
            tau: Attack probability threshold for last-hit suggestions
            voice: macOS voice name for speech synthesis
        """
        self.gsi_dir = Path(gsi_dir)
        self.tau = tau
        self.voice = voice
        
        # Load policy if provided
        self.policy = None
        if policy_path:
            try:
                self.policy = torch.load(policy_path, map_location='cpu')
                self.policy.eval()
                print(f"📦 Loaded policy from {policy_path}")
            except Exception as e:
                print(f"⚠️  Failed to load policy: {e}")
        
        # Initialize prompt bus
        stats_file = self.gsi_dir / "coach_stats.json"
        self.prompt_bus = PromptBus(
            max_rate_per_min=6,
            cooldown_s=6.0,
            wilson_alpha=0.05,
            min_lb=0.55,
            stats_file=str(stats_file)
        )
        
        # File monitoring state
        self.current_file: Optional[Path] = None
        self.file_position = 0
        self.last_processed_time = 0.0
    
    def find_newest_session_file(self) -> Optional[Path]:
        """Find the newest GSI session file."""
        session_files = list(self.gsi_dir.glob("session-*.ndjson"))
        if not session_files:
            return None
        
        # Sort by modification time, newest first
        return max(session_files, key=lambda f: f.stat().st_mtime)
    
    def process_new_lines(self, file_path: Path) -> int:
        """
        Process new lines from the session file.
        
        Returns:
            Number of new game events processed
        """
        events_processed = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Seek to last known position
                f.seek(self.file_position)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        log_entry = json.loads(line)
                        
                        # Skip session_start records
                        if log_entry.get("type") == "session_start":
                            continue
                        
                        # Process game data
                        if "data" in log_entry:
                            timestamp = log_entry["ts"]
                            
                            # Avoid reprocessing old events
                            if timestamp <= self.last_processed_time:
                                continue
                            
                            game_data = log_entry["data"]
                            
                            # Parse into GSISnapshot
                            snapshot = parse_gsi_payload(game_data, timestamp)
                            
                            # Generate and potentially emit prompts
                            self.process_snapshot(snapshot, timestamp)
                            
                            events_processed += 1
                            self.last_processed_time = timestamp
                    
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
                    except Exception as e:
                        print(f"⚠️  Error processing line: {e}")
                        continue
                
                # Update file position
                self.file_position = f.tell()
        
        except IOError as e:
            print(f"⚠️  Error reading file: {e}")
        
        return events_processed
    
    def process_snapshot(self, snapshot: GSISnapshot, timestamp: float):
        """Process a game state snapshot and potentially emit coaching prompts."""
        # Generate coaching prompts
        prompts = get_coaching_prompts(snapshot, self.policy, self.tau)
        
        now = time.time()
        
        for key, text in prompts.items():
            # For now, assume all prompts are potentially helpful
            # In a real system, this would be based on user feedback or game outcome analysis
            evidence = True
            
            # Try to emit through prompt bus
            if self.prompt_bus.maybe_emit(key, text, evidence, now):
                print(f"🎯 [{timestamp:.1f}s] {text}")
                
                # Speak the prompt
                if speak(text, voice=self.voice):
                    print(f"🔊 Spoken: {text}")
                else:
                    print(f"⚠️  Failed to speak: {text}")
    
    def monitor_and_coach(self, check_interval: float = 1.0):
        """
        Main monitoring loop that watches for new GSI data and provides coaching.
        
        Args:
            check_interval: How often to check for updates (seconds)
        """
        print(f"🎮 Starting live coaching system")
        print(f"📁 Monitoring: {self.gsi_dir}")
        print(f"🔊 Voice: {self.voice or 'system default'}")
        print(f"⏱️  Check interval: {check_interval}s")
        print("🚨 Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while True:
                # Find newest session file
                newest_file = self.find_newest_session_file()
                
                if newest_file is None:
                    print("⏳ Waiting for GSI session file...")
                    time.sleep(check_interval * 2)
                    continue
                
                # Switch to new file if needed
                if self.current_file != newest_file:
                    print(f"📂 Switching to: {newest_file.name}")
                    self.current_file = newest_file
                    self.file_position = 0
                    self.last_processed_time = 0.0
                
                # Process any new lines
                events_processed = self.process_new_lines(self.current_file)
                
                if events_processed > 0:
                    print(f"📊 Processed {events_processed} new events")
                
                # Wait before next check
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping live coaching system")
            self.print_stats()
    
    def print_stats(self):
        """Print coaching statistics."""
        stats = self.prompt_bus.get_stats()
        
        if not stats:
            print("📊 No coaching statistics available")
            return
        
        print("\n📈 COACHING STATISTICS")
        print("-" * 40)
        
        for key, data in stats.items():
            gated = " (GATED)" if data["gated"] else ""
            print(f"{key:<12}: {data['wins']}/{data['total']} "
                  f"(LB: {data['wilson_lb']:.3f}){gated}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live Dota 2 coaching system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--gsi-dir",
        default="artifacts/gsi",
        help="Directory containing GSI session logs"
    )
    
    parser.add_argument(
        "--policy",
        help="Path to PyTorch policy file (.pt)"
    )
    
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help="Attack probability threshold for last-hit suggestions"
    )
    
    parser.add_argument(
        "--voice",
        help="macOS voice name (e.g., 'Alex', 'Samantha')"
    )
    
    parser.add_argument(
        "--check-interval",
        type=float,
        default=1.0,
        help="How often to check for file updates (seconds)"
    )
    
    args = parser.parse_args()
    
    gsi_dir = Path(args.gsi_dir)
    if not gsi_dir.exists():
        print(f"❌ GSI directory not found: {gsi_dir}")
        return 1
    
    # Create live coach
    coach = LiveCoach(
        gsi_dir=str(gsi_dir),
        policy_path=args.policy,
        tau=args.tau,
        voice=args.voice
    )
    
    # Start monitoring
    coach.monitor_and_coach(args.check_interval)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())