#!/usr/bin/env python3
"""
Multi-agent policy and value networks with per-role observation dimensions.
"""

import torch
import torch.nn as nn
from typing import Dict


class DimAdapter(nn.Module):
    """
    Dimension adapter to handle observation dimension mismatches.
    Used when loading checkpoints with different obs dims than current environment.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MultiHeadPolicy(nn.Module):
    """
    Multi-head policy network with separate heads for different agent roles.
    Each role gets its own input layer to handle different observation dimensions.
    Uses role-keyed ModuleDicts for safe checkpoint loading.
    """
    
    def __init__(self, obs_dims: Dict[str, int], n_actions: int = 5):
        super().__init__()
        self.obs_dims = obs_dims
        self.n_actions = n_actions
        
        # Create role-keyed ModuleDict for policy heads
        self.pi = nn.ModuleDict()
        self.vf = nn.ModuleDict()
        
        for role, obs_dim in obs_dims.items():
            self.pi[role] = PolicyHead(obs_dim, n_actions)
            self.vf[role] = ValueHead(obs_dim)
    
    def act(self, role: str, obs: torch.Tensor) -> torch.Tensor:
        """Get policy logits for specific role."""
        if role not in self.pi:
            raise ValueError(f"Unknown role: {role}. Available: {list(self.pi.keys())}")
        return self.pi[role](obs)
    
    def value(self, role: str, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate for specific role."""
        if role not in self.vf:
            raise ValueError(f"Unknown role: {role}. Available: {list(self.vf.keys())}")
        return self.vf[role](obs)
    
    def forward(self, role: str, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for specific role (backward compatibility)."""
        return self.act(role, obs)


class MultiHeadValue(nn.Module):
    """
    Multi-head value network with separate heads for different agent roles.
    Each role gets its own input layer to handle different observation dimensions.
    """
    
    def __init__(self, obs_dims: Dict[str, int]):
        super().__init__()
        self.obs_dims = obs_dims
        
        # Create separate value heads for each role
        self.heads = nn.ModuleDict()
        for role, obs_dim in obs_dims.items():
            self.heads[role] = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.Tanh(), 
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Optional adapters for dimension mismatches
        self.adapters = nn.ModuleDict()
    
    def forward(self, role: str, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for specific role."""
        if role not in self.heads:
            raise ValueError(f"Unknown role: {role}. Available: {list(self.heads.keys())}")
        
        # Apply adapter if needed
        if role in self.adapters:
            obs = self.adapters[role](obs)
        
        return self.heads[role](obs).squeeze(-1)
    
    def add_adapter(self, role: str, env_dim: int, ckpt_dim: int):
        """Add dimension adapter for a specific role."""
        if env_dim != ckpt_dim:
            self.adapters[role] = DimAdapter(env_dim, ckpt_dim)
            print(f"WARNING: Added adapter for role '{role}': env_dim={env_dim} -> ckpt_dim={ckpt_dim}")
    
    def get_role_head(self, role: str) -> nn.Module:
        """Get the value head for a specific role."""
        return self.heads[role]


class PolicyHead(nn.Module):
    """
    Legacy single-role policy head for backwards compatibility.
    """
    
    def __init__(self, in_dim: int, n_act: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, n_act)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueHead(nn.Module):
    """
    Legacy single-role value head for backwards compatibility.
    """
    
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)