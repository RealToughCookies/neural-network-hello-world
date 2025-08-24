"""
Prioritized opponent pool with JSON persistence for RL training.
"""
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class OpponentPool:
    """
    Manages a pool of opponent checkpoints with EMA win rates and prioritized sampling.
    
    Persists state to JSON file with structure per opponent:
    {
        "ckpt": "artifacts/rl_ckpts/step_XXXX.pt",
        "wins": int,
        "games": int, 
        "ema_wr": float,
        "last_seen_step": int
    }
    """
    
    def __init__(self, path: Union[str, Path], max_size: int = 64, 
                 ema_decay: float = 0.97, min_games: int = 5):
        """
        Initialize opponent pool.
        
        Args:
            path: JSON file path for persistence
            max_size: Maximum number of opponents to keep
            ema_decay: EMA decay factor for win rate (higher = more history)
            min_games: Minimum games before using EMA win rate for sampling
        """
        self.path = Path(path)
        self.max_size = max_size
        self.ema_decay = ema_decay
        self.min_games = min_games
        self.opponents: Dict[str, Dict] = {}
        
        # Create directory if needed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing pool
        self._load()
    
    def _load(self) -> None:
        """Load opponents from JSON file."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    self.opponents = data.get('opponents', {})
                logger.info(f"Loaded {len(self.opponents)} opponents from {self.path}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load opponent pool from {self.path}: {e}")
                self.opponents = {}
        else:
            logger.info(f"Creating new opponent pool at {self.path}")
            self.opponents = {}
    
    def _save(self) -> None:
        """Save opponents to JSON file."""
        try:
            data = {'opponents': self.opponents}
            with open(self.path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.opponents)} opponents to {self.path}")
        except IOError as e:
            logger.error(f"Failed to save opponent pool to {self.path}: {e}")
    
    def add(self, ckpt_path: Union[str, Path], step: int) -> None:
        """
        Add or refresh an opponent checkpoint.
        
        Args:
            ckpt_path: Path to checkpoint file
            step: Training step when checkpoint was created
        """
        ckpt_str = str(Path(ckpt_path).resolve())
        
        if ckpt_str in self.opponents:
            # Update existing opponent
            self.opponents[ckpt_str]['last_seen_step'] = step
            logger.debug(f"Updated opponent {ckpt_str} last_seen_step to {step}")
        else:
            # Add new opponent
            self.opponents[ckpt_str] = {
                'ckpt': ckpt_str,
                'wins': 0,
                'games': 0,
                'ema_wr': 0.5,  # Start at neutral
                'last_seen_step': step
            }
            logger.info(f"Added new opponent {ckpt_str} at step {step}")
        
        # Prune if needed
        if len(self.opponents) > self.max_size:
            self.prune(self.max_size)
        
        self._save()
    
    def record_result(self, ckpt_path: Union[str, Path], won: bool) -> None:
        """
        Record game result against an opponent.
        
        Args:
            ckpt_path: Path to opponent checkpoint
            won: True if we won, False if we lost
        """
        ckpt_str = str(Path(ckpt_path).resolve())
        
        if ckpt_str not in self.opponents:
            logger.warning(f"Recording result for unknown opponent {ckpt_str}")
            return
        
        opp = self.opponents[ckpt_str]
        opp['games'] += 1
        if won:
            opp['wins'] += 1
        
        # Update EMA win rate
        win_rate = 1.0 if won else 0.0
        opp['ema_wr'] = self.ema_decay * opp['ema_wr'] + (1 - self.ema_decay) * win_rate
        
        logger.debug(f"Recorded {'win' if won else 'loss'} vs {ckpt_str}: "
                    f"{opp['wins']}/{opp['games']} games, ema_wr={opp['ema_wr']:.3f}")
        
        # Save after every result to avoid data loss
        self._save()
    
    def decay_all(self, decay_factor: float = 0.99) -> None:
        """
        Apply slight decay of all EMA win rates toward 0.5 to prevent lock-in.
        
        Args:
            decay_factor: How much to decay toward neutral (0.5)
        """
        for opp in self.opponents.values():
            # Decay toward 0.5
            opp['ema_wr'] = decay_factor * opp['ema_wr'] + (1 - decay_factor) * 0.5
        
        if self.opponents:
            logger.debug(f"Applied decay to {len(self.opponents)} opponents")
            self._save()
    
    def sample_uniform(self, rng: np.random.Generator) -> Optional[str]:
        """
        Sample opponent uniformly at random.
        
        Args:
            rng: Numpy random generator
            
        Returns:
            Checkpoint path of selected opponent, or None if pool empty
        """
        if not self.opponents:
            return None
        
        ckpt_paths = list(self.opponents.keys())
        idx = rng.integers(0, len(ckpt_paths))
        return ckpt_paths[idx]
    
    def sample_prioritized(self, rng: np.random.Generator, temp: float = 0.7) -> Optional[str]:
        """
        Sample opponent using prioritized sampling based on win rate.
        
        Args:
            rng: Numpy random generator
            temp: Temperature for softmax (lower = more focused on difficult opponents)
            
        Returns:
            Checkpoint path of selected opponent, or None if pool empty
        """
        if not self.opponents:
            return None
        
        ckpt_paths = list(self.opponents.keys())
        logits = []
        
        for ckpt_path in ckpt_paths:
            opp = self.opponents[ckpt_path]
            
            # Use EMA win rate if enough games, otherwise neutral
            if opp['games'] >= self.min_games:
                score = np.clip(opp['ema_wr'], 0.05, 0.95)
            else:
                score = 0.5
            
            # Convert to logit: log(p / (1-p)) / temp
            logit = math.log(score / (1 - score)) / temp
            logits.append(logit)
        
        # Softmax to get probabilities
        logits = np.array(logits)
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample
        idx = rng.choice(len(ckpt_paths), p=probs)
        selected = ckpt_paths[idx]
        
        logger.debug(f"Prioritized sample: {selected} (ema_wr={self.opponents[selected]['ema_wr']:.3f})")
        return selected
    
    def prune(self, max_size: int) -> None:
        """
        Prune pool to max_size, keeping most recent and diverse opponents.
        
        Sort by:
        1. last_seen_step (desc) - keep recent
        2. |ema_wr - 0.5| (asc) - keep diverse difficulties
        
        Args:
            max_size: Maximum opponents to keep
        """
        if len(self.opponents) <= max_size:
            return
        
        # Sort opponents for pruning
        items = list(self.opponents.items())
        
        def sort_key(item):
            ckpt_path, opp = item
            recency = -opp['last_seen_step']  # Negative for desc sort
            difficulty_diversity = abs(opp['ema_wr'] - 0.5)  # Closer to 0.5 = more diverse
            return (recency, difficulty_diversity)
        
        items.sort(key=sort_key)
        
        # Keep top max_size opponents
        kept_items = items[:max_size]
        pruned_count = len(items) - max_size
        
        self.opponents = dict(kept_items)
        
        logger.info(f"Pruned {pruned_count} opponents, kept {len(self.opponents)}")
        self._save()
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        if not self.opponents:
            return {'size': 0}
        
        ema_wrs = [opp['ema_wr'] for opp in self.opponents.values()]
        games = [opp['games'] for opp in self.opponents.values()]
        
        return {
            'size': len(self.opponents),
            'avg_ema_wr': np.mean(ema_wrs),
            'std_ema_wr': np.std(ema_wrs),
            'min_ema_wr': np.min(ema_wrs),
            'max_ema_wr': np.max(ema_wrs),
            'total_games': np.sum(games),
            'avg_games': np.mean(games)
        }
    
    def __len__(self) -> int:
        """Return number of opponents in pool."""
        return len(self.opponents)
    
    def __bool__(self) -> bool:
        """Return True if pool is non-empty."""
        return len(self.opponents) > 0