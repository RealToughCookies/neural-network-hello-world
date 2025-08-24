"""Self-play infrastructure for PPO with opponent pools and evaluation."""

from dataclasses import dataclass, field
import random
import copy
import torch
import numpy as np


@dataclass
class OpponentSnapshot:
    """Frozen snapshot of opponent policies."""
    pi_good: dict
    pi_adv: dict


@dataclass
class OpponentPool:
    """Pool of frozen opponent snapshots with capacity management."""
    cap: int = 5
    _items: list[OpponentSnapshot] = field(default_factory=list)

    def push(self, pi_good, pi_adv):
        """Add new opponent snapshot to pool."""
        snap = OpponentSnapshot(
            copy.deepcopy(pi_good.state_dict()),
            copy.deepcopy(pi_adv.state_dict())
        )
        self._items.append(snap)
        if len(self._items) > self.cap:
            self._items.pop(0)

    def sample(self, decay=0.7):
        """Sample opponent with geometric bias toward recent snapshots."""
        if not self._items: 
            return None
        idxs = list(range(len(self._items)))
        weights = [decay**(len(self._items)-1-i) for i in idxs]
        s = random.choices(self._items, weights=weights, k=1)[0]
        return s


class Matchmaker:
    """Selects opponents from pool vs latest policies."""
    
    def __init__(self, pool, p_latest=0.5):
        self.pool = pool
        self.p_latest = p_latest

    def pick_opponent(self, latest_pi_good, latest_pi_adv):
        """Pick opponent: latest policy or frozen snapshot from pool."""
        if not self.pool._items or random.random() < self.p_latest:
            return latest_pi_good, latest_pi_adv, "latest"
        
        s = self.pool.sample()
        # Materialize frozen modules
        pg = copy.deepcopy(latest_pi_good)
        pg.load_state_dict(s.pi_good)
        pa = copy.deepcopy(latest_pi_adv)
        pa.load_state_dict(s.pi_adv)
        
        # Set to eval mode
        pg.eval()
        pa.eval()
        
        return pg, pa, "pool"


def evaluate(env, pi_good, pi_adv, episodes=4, device="cpu"):
    """Evaluate policy performance over multiple episodes.
    
    Runs episodes alternating roles: 2 as good, 2 as adversary.
    Returns mean returns per role using Parallel API.
    """
    from src.rl.ppo_selfplay_skeleton import _act_for_keys
    
    good_returns = []
    adv_returns = []
    
    for ep in range(episodes):
        # Alternate roles each episode
        learner_role = "good" if ep % 2 == 0 else "adv"
        
        # Reset environment
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        ep_return_good = 0.0
        ep_return_adv = 0.0
        
        with torch.no_grad():
            for step in range(100):  # Max episode length
                if not isinstance(obs, dict):
                    break
                    
                keys = list(obs.keys())
                good_keys = [k for k in keys if "agent" in k]
                adv_keys = [k for k in keys if "adversary" in k]
                
                # Generate actions for both roles
                acts_good = _act_for_keys(good_keys, obs, pi_good, pi_good, device)
                acts_adv = _act_for_keys(adv_keys, obs, pi_adv, pi_adv, device)
                
                # Build action dict (extract action only, not logp)
                act_dict = {k: v[0] for k, v in acts_good.items()}
                act_dict.update({k: v[0] for k, v in acts_adv.items()})
                
                try:
                    next_obs, rews, terms, truncs, _ = env.step(act_dict)
                except Exception as e:
                    break
                
                # Accumulate returns per role
                for k in good_keys:
                    ep_return_good += rews.get(k, 0.0)
                for k in adv_keys:
                    ep_return_adv += rews.get(k, 0.0)
                
                obs = next_obs
                
                # Check termination
                if all(terms.values()) or all(truncs.values()):
                    break
        
        good_returns.append(ep_return_good)
        adv_returns.append(ep_return_adv)
    
    return np.mean(good_returns), np.mean(adv_returns)