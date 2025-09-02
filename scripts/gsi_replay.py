#!/usr/bin/env python3
"""
Replay and analyze Dota 2 GSI logs to generate coaching prompts.

Usage:
    python -m scripts.gsi_replay --log artifacts/gsi/session-*.ndjson --max-rate 6
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gsi.schema import (
    get_health_percent, get_gold, has_boots, has_tp_scroll, 
    get_ready_abilities, safe_get, get_game_time
)


class RateLimiter:
    """Simple rate limiter for coaching prompts."""
    
    def __init__(self, max_rate: int = 6):
        """Initialize with max prompts per minute."""
        self.max_rate = max_rate
        self.prompt_times: List[float] = []
    
    def can_prompt(self) -> bool:
        """Check if we can send another prompt."""
        now = time.time()
        # Remove prompts older than 1 minute
        self.prompt_times = [t for t in self.prompt_times if now - t < 60.0]
        
        return len(self.prompt_times) < self.max_rate
    
    def record_prompt(self):
        """Record that we sent a prompt."""
        self.prompt_times.append(time.time())


def check_low_health(data: Dict[str, Any]) -> str:
    """Check if hero has dangerously low health."""
    health_pct = get_health_percent(data)
    if health_pct is not None and health_pct < 0.35:
        return "Back, low HP."
    return None


def check_ready_abilities(data: Dict[str, Any]) -> str:
    """Check if key abilities are ready to use."""
    ready_abilities = get_ready_abilities(data)
    if ready_abilities:
        return "Key ability up."
    return None


def check_tp_scroll(data: Dict[str, Any]) -> str:
    """Check if player needs to buy TP scrolls."""
    if not has_tp_scroll(data):
        return "Buy TP."
    return None


def check_boots_purchase(data: Dict[str, Any]) -> str:
    """Check if player should consider buying boots."""
    gold = get_gold(data)
    has_boots_item = has_boots(data)
    
    if gold is not None and gold >= 500 and not has_boots_item:
        return "Consider Boots."
    return None


def analyze_game_state(data: Dict[str, Any]) -> List[str]:
    """
    Analyze game state and return list of coaching prompts.
    
    Implements the naive coaching rules:
    - hero.health_percent < 0.35 → "Back, low HP."
    - any(abilities.*.cooldown == 0 and .level > 0) → "Key ability up."
    - no TP item in items → "Buy TP."
    - player.gold >= 500 and no boots → "Consider Boots."
    """
    prompts = []
    
    # Check each condition
    checks = [
        check_low_health,
        check_ready_abilities,
        check_tp_scroll,
        check_boots_purchase
    ]
    
    for check_func in checks:
        result = check_func(data)
        if result:
            prompts.append(result)
    
    return prompts


def process_log_file(log_path: Path, rate_limiter: RateLimiter, verbose: bool = False):
    """Process a single NDJSON log file and generate coaching prompts."""
    
    print(f"📖 Processing: {log_path.name}")
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_path}")
        return
    
    prompt_count = 0
    event_count = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse NDJSON line
                    log_entry = json.loads(line)
                    data = log_entry.get('data', {})
                    timestamp = log_entry.get('ts', 0)
                    
                    event_count += 1
                    
                    # Get game time for context
                    game_time = get_game_time(data)
                    game_time_str = f"{game_time:.1f}s" if game_time is not None else "??s"
                    
                    # Analyze the game state
                    prompts = analyze_game_state(data)
                    
                    # Send rate-limited prompts
                    for prompt in prompts:
                        if rate_limiter.can_prompt():
                            rate_limiter.record_prompt()
                            prompt_count += 1
                            
                            print(f"🎯 [{game_time_str}] {prompt}")
                        else:
                            if verbose:
                                print(f"⏱️  Rate limited: {prompt}")
                    
                    # Show debug info for first few events
                    if verbose and event_count <= 3:
                        health = get_health_percent(data)
                        gold = get_gold(data)
                        boots = has_boots(data)
                        tp = has_tp_scroll(data)
                        abilities = get_ready_abilities(data)
                        
                        health_str = f"{health:.2f}" if health is not None else "None"
                        print(f"   Debug: HP={health_str}, "
                              f"Gold={gold}, Boots={boots}, TP={tp}, "
                              f"Ready abilities={abilities}")
                
                except json.JSONDecodeError as e:
                    print(f"⚠️  Line {line_num}: Invalid JSON - {e}")
                    continue
                
                except Exception as e:
                    print(f"⚠️  Line {line_num}: Error processing - {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return
    
    print(f"✅ Processed {event_count} events, sent {prompt_count} coaching prompts")


def main():
    """Main entry point for GSI replay analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze Dota 2 GSI logs and generate coaching prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--log",
        required=True,
        help="Path to NDJSON log file (supports wildcards like 'session-*.ndjson')"
    )
    
    parser.add_argument(
        "--max-rate",
        type=int,
        default=6,
        help="Maximum prompts per minute (rate limiting)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show debug information and rate-limited prompts"
    )
    
    args = parser.parse_args()
    
    # Handle glob patterns in log path
    log_path = Path(args.log)
    
    if '*' in str(log_path):
        # Handle wildcards
        parent_dir = log_path.parent
        pattern = log_path.name
        matching_files = list(parent_dir.glob(pattern))
        
        if not matching_files:
            print(f"❌ No files found matching: {args.log}")
            return 1
        
        log_files = sorted(matching_files)
        print(f"📁 Found {len(log_files)} matching files")
    else:
        log_files = [log_path]
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(max_rate=args.max_rate)
    
    print(f"🎮 Starting GSI log analysis (max {args.max_rate} prompts/min)")
    print("=" * 60)
    
    # Process each log file
    for log_file in log_files:
        process_log_file(log_file, rate_limiter, args.verbose)
        if len(log_files) > 1:
            print("-" * 40)
    
    print("=" * 60)
    print("✅ Analysis complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())