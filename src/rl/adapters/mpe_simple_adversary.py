"""
MPE Simple Adversary environment adapter implementation.
Wraps PettingZoo simple_adversary_v3.parallel_env with standardized API.
"""

from typing import Dict, List, Any
import numpy as np
from src.rl.env_api import register, Timestep

@register("mpe_adversary")
class MPESimpleAdversaryAdapter:
    """Adapter for PettingZoo MPE simple_adversary_v3 environment."""
    
    def __init__(self, render_mode: str | None = None, continuous_actions: bool = False):
        """
        Initialize MPE Simple Adversary adapter.
        
        Args:
            render_mode: Rendering mode for visualization (None, "human", etc.)
            continuous_actions: Whether to use continuous action space
        """
        # Import PettingZoo only when needed
        from pettingzoo.mpe import simple_adversary_v3
        
        self.env = simple_adversary_v3.parallel_env(
            render_mode=render_mode,
            continuous_actions=continuous_actions
        )
        
        # Cache agent information
        self._agent_names = list(self.env.possible_agents)
        self._roles = self._compute_roles()
        self._obs_dims = self._compute_obs_dims()
        self._n_actions = self._compute_n_actions()
        
        print(f"[MPE adapter] {len(self._agent_names)} agents: {self._agent_names}")
        print(f"[MPE adapter] roles: {self._roles}")
        print(f"[MPE adapter] obs_dims: {self._obs_dims}")
        print(f"[MPE adapter] n_actions: {self._n_actions}")
    
    def _compute_roles(self) -> Dict[str, str]:
        """Compute agent name to role mapping."""
        roles = {}
        for agent_name in self._agent_names:
            if "adversary" in agent_name:
                roles[agent_name] = "adv"
            else:
                roles[agent_name] = "good"
        return roles
    
    def _compute_obs_dims(self) -> Dict[str, int]:
        """Compute role to observation dimension mapping."""
        obs_dims = {}
        
        for agent_name in self._agent_names:
            role = self._roles[agent_name]
            agent_obs_dim = self.env.observation_space(agent_name).shape[0]
            
            if role not in obs_dims:
                obs_dims[role] = agent_obs_dim
            elif obs_dims[role] != agent_obs_dim:
                raise ValueError(
                    f"Inconsistent obs dims within role {role}: "
                    f"expected {obs_dims[role]}, got {agent_obs_dim} for {agent_name}"
                )
        
        return obs_dims
    
    def _compute_n_actions(self) -> int:
        """Compute number of discrete actions (must be same for all agents)."""
        n_actions = None
        
        for agent_name in self._agent_names:
            agent_n_actions = self.env.action_space(agent_name).n
            
            if n_actions is None:
                n_actions = agent_n_actions
            elif n_actions != agent_n_actions:
                raise ValueError(
                    f"Inconsistent action space sizes: "
                    f"expected {n_actions}, got {agent_n_actions} for {agent_name}"
                )
        
        return n_actions
    
    def reset(self, seed: int | None = None) -> Timestep:
        """Reset environment and return initial timestep."""
        if seed is not None:
            # Set seed on environment if it supports it
            try:
                obs, infos = self.env.reset(seed=seed)
            except TypeError:
                # Fallback for environments that don't support seed parameter
                obs, infos = self.env.reset()
        else:
            obs, infos = self.env.reset()
        
        # Convert observations to float32
        for agent_name in obs:
            obs[agent_name] = obs[agent_name].astype(np.float32)
        
        # Ensure all expected agents have entries
        for agent_name in self._agent_names:
            if agent_name not in obs:
                obs[agent_name] = np.zeros(self._obs_dims[self._roles[agent_name]], dtype=np.float32)
            if agent_name not in infos:
                infos[agent_name] = {}
        
        # Initial timestep has no rewards or dones
        rewards = {agent_name: 0.0 for agent_name in self._agent_names}
        dones = {agent_name: False for agent_name in self._agent_names}
        
        return Timestep(obs=obs, rewards=rewards, dones=dones, infos=infos)
    
    def step(self, action_by_agent: Dict[str, int]) -> Timestep:
        """Step environment with actions and return next timestep."""
        # Convert actions to env order and filter to active agents
        env_actions = {}
        for agent_name in self.env.agents:
            if agent_name in action_by_agent:
                env_actions[agent_name] = action_by_agent[agent_name]
        
        # Step the environment
        obs, rewards, dones, truncated, infos = self.env.step(env_actions)
        
        # Convert observations to float32 numpy arrays
        for agent_name in obs:
            obs[agent_name] = obs[agent_name].astype(np.float32)
        
        # Combine done and truncated into single done signal
        for agent_name in dones:
            if agent_name in truncated:
                dones[agent_name] = dones[agent_name] or truncated[agent_name]
        
        # Ensure all expected agents have entries (fill with defaults if missing)
        for agent_name in self._agent_names:
            if agent_name not in obs:
                obs[agent_name] = np.zeros(self._obs_dims[self._roles[agent_name]], dtype=np.float32)
            if agent_name not in rewards:
                rewards[agent_name] = 0.0
            if agent_name not in dones:
                dones[agent_name] = True  # Agent is done if not in active agents
            if agent_name not in infos:
                infos[agent_name] = {}
        
        return Timestep(obs=obs, rewards=rewards, dones=dones, infos=infos)
    
    def close(self) -> None:
        """Close environment and clean up resources."""
        self.env.close()
    
    def roles(self) -> Dict[str, str]:
        """Return mapping from agent_name to role."""
        return self._roles.copy()
    
    def obs_dims(self) -> Dict[str, int]:
        """Return mapping from role to observation dimension."""
        return self._obs_dims.copy()
    
    def n_actions(self) -> int:
        """Return number of discrete actions."""
        return self._n_actions
    
    def agent_names(self) -> List[str]:
        """Return list of agent names in consistent order."""
        return self._agent_names.copy()