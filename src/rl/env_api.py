from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple, Any
import numpy as np

@dataclass
class Timestep:
    """Standardized timestep representation for multi-agent environments."""
    obs: Dict[str, np.ndarray]
    rewards: Dict[str, float] 
    dones: Dict[str, bool]
    infos: Dict[str, Any]

class EnvAdapter(Protocol):
    """Protocol for environment adapters that abstract away specific environment APIs."""
    
    def reset(self, seed: int | None = None) -> Timestep:
        """Reset environment and return initial timestep."""
        ...
    
    def step(self, action_by_agent: Dict[str, int]) -> Timestep:
        """Step environment with actions and return next timestep."""
        ...
    
    def close(self) -> None:
        """Close environment and clean up resources."""
        ...
    
    def roles(self) -> Dict[str, str]:
        """Return mapping from agent_name to role (e.g. "good", "adv")."""
        ...
    
    def obs_dims(self) -> Dict[str, int]:
        """Return mapping from role to observation dimension."""
        ...
    
    def n_actions(self) -> int:
        """Return number of discrete actions (shared across all agents)."""
        ...
    
    def agent_names(self) -> List[str]:
        """Return list of agent names in consistent order."""
        ...

# Global registry for environment adapters
_REGISTRY: Dict[str, type] = {}

def register(name: str):
    """Decorator to register an environment adapter."""
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap

def make_adapter(name: str, **kwargs) -> EnvAdapter:
    """Create an environment adapter by name."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown env adapter: {name}. Available: {available}")
    return _REGISTRY[name](**kwargs)