"""
Rate-limited prompt bus with Wilson confidence interval gating.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def wilson_score_interval_lower(wins: int, total: int, alpha: float = 0.05) -> float:
    """
    Calculate Wilson score interval lower bound.
    
    Based on Evan Miller's formula:
    https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
    
    Args:
        wins: Number of positive outcomes
        total: Total number of trials
        alpha: Significance level (0.05 for 95% confidence)
        
    Returns:
        Lower bound of Wilson confidence interval
    """
    if total == 0:
        return 0.0
    
    z = 1.96  # 97.5th percentile of standard normal for alpha=0.05
    p_hat = wins / total
    
    numerator = (
        p_hat + (z * z) / (2 * total) - 
        z * math.sqrt((p_hat * (1 - p_hat) + (z * z) / (4 * total)) / total)
    )
    denominator = 1 + (z * z) / total
    
    return numerator / denominator


class PromptBus:
    """
    Rate-limited prompt bus with per-key cooldowns and Wilson CI gating.
    """
    
    def __init__(self, max_rate_per_min: int = 6, cooldown_s: float = 6.0, 
                 wilson_alpha: float = 0.05, min_lb: float = 0.55, 
                 stats_file: Optional[str] = None):
        """
        Initialize prompt bus.
        
        Args:
            max_rate_per_min: Maximum prompts per minute globally
            cooldown_s: Minimum seconds between prompts of same key
            wilson_alpha: Alpha for Wilson confidence interval (0.05 = 95% CI)
            min_lb: Minimum Wilson lower bound required to emit prompt
            stats_file: Path to JSON file for storing win/loss stats
        """
        self.max_rate_per_min = max_rate_per_min
        self.cooldown_s = cooldown_s
        self.wilson_alpha = wilson_alpha
        self.min_lb = min_lb
        
        # Rate limiting state
        self.prompt_times: List[float] = []
        self.last_prompt_by_key: Dict[str, float] = {}
        
        # Statistics tracking
        self.stats_file = Path(stats_file) if stats_file else Path("artifacts/coach/prompt_stats.json")
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict[str, Dict[str, int]]:
        """Load win/loss statistics from JSON file."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {}
    
    def _save_stats(self):
        """Save win/loss statistics to JSON file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except IOError:
            pass  # Fail silently to avoid breaking the coaching flow
    
    def _check_rate_limit(self, now: float) -> bool:
        """Check if global rate limit allows new prompt."""
        # Remove prompts older than 1 minute
        cutoff = now - 60.0
        self.prompt_times = [t for t in self.prompt_times if t > cutoff]
        
        return len(self.prompt_times) < self.max_rate_per_min
    
    def _check_cooldown(self, key: str, now: float) -> bool:
        """Check if per-key cooldown allows new prompt."""
        last_time = self.last_prompt_by_key.get(key, 0)
        return (now - last_time) >= self.cooldown_s
    
    def _check_wilson_gate(self, key: str) -> bool:
        """Check if Wilson lower bound meets minimum threshold."""
        if key not in self.stats:
            return True  # Allow first few prompts to establish baseline
        
        wins = self.stats[key].get("wins", 0)
        total = self.stats[key].get("total", 0)
        
        if total < 3:  # Need minimum samples for meaningful CI
            return True
        
        lower_bound = wilson_score_interval_lower(wins, total, self.wilson_alpha)
        return lower_bound >= self.min_lb
    
    def maybe_emit(self, key: str, text: str, evidence: bool, now: float) -> bool:
        """
        Try to emit a prompt if rate limits and quality gates pass.
        
        Args:
            key: Unique key for this prompt type (e.g., "retreat", "tp_scroll")
            text: Prompt text to potentially emit
            evidence: Whether the prompt was actually helpful (for learning)
            now: Current timestamp
            
        Returns:
            True if prompt was emitted, False if suppressed
        """
        # Check rate limit
        if not self._check_rate_limit(now):
            return False
        
        # Check per-key cooldown
        if not self._check_cooldown(key, now):
            return False
        
        # Check Wilson confidence interval gate
        if not self._check_wilson_gate(key):
            return False
        
        # All gates passed - emit the prompt
        self.prompt_times.append(now)
        self.last_prompt_by_key[key] = now
        
        # Update statistics (this would normally be done asynchronously based on user feedback)
        # For now, we assume evidence reflects immediate usefulness
        if key not in self.stats:
            self.stats[key] = {"wins": 0, "total": 0}
        
        self.stats[key]["total"] += 1
        if evidence:
            self.stats[key]["wins"] += 1
        
        self._save_stats()
        
        return True
    
    def update_feedback(self, key: str, was_helpful: bool):
        """
        Update statistics based on user feedback.
        
        Args:
            key: Prompt type key
            was_helpful: Whether the prompt was actually helpful
        """
        if key not in self.stats:
            self.stats[key] = {"wins": 0, "total": 0}
        
        self.stats[key]["total"] += 1
        if was_helpful:
            self.stats[key]["wins"] += 1
        
        self._save_stats()
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get current statistics with Wilson confidence intervals.
        
        Returns:
            Dict mapping keys to stats including win rate and Wilson lower bound
        """
        result = {}
        
        for key, stats in self.stats.items():
            wins = stats.get("wins", 0)
            total = stats.get("total", 0)
            
            win_rate = wins / total if total > 0 else 0.0
            lower_bound = wilson_score_interval_lower(wins, total, self.wilson_alpha)
            
            result[key] = {
                "wins": wins,
                "total": total,
                "win_rate": win_rate,
                "wilson_lb": lower_bound,
                "gated": lower_bound < self.min_lb and total >= 3
            }
        
        return result