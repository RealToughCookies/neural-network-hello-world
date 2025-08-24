#!/usr/bin/env python3
"""
Environment utilities for handling multi-agent role mapping and observation dimensions.
"""

from typing import Dict, Tuple


def get_role_maps(env) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Extract role mapping and observation dimensions from PettingZoo environment.
    
    Args:
        env: PettingZoo environment instance
        
    Returns:
        role_of: maps agent_name -> role ("good" or "adv")  
        obs_dim: maps role -> obs_dim (int)
        
    Assumes MPE simple_adversary_v3 structure:
    - One "adversary_*" agent -> "adv" role
    - Multiple "agent_*" agents -> "good" role
    """
    role_of = {}
    obs_dim = {}
    
    # Map each agent to its role
    for agent_name in env.possible_agents:
        role = "adv" if "adversary" in agent_name else "good"
        role_of[agent_name] = role
        
        # Get observation dimension for this agent
        agent_obs_dim = env.observation_space(agent_name).shape[0]
        
        # Store the obs dimension for this role (should be consistent within role)
        if role not in obs_dim:
            obs_dim[role] = agent_obs_dim
        elif obs_dim[role] != agent_obs_dim:
            raise ValueError(f"Inconsistent obs dims within role {role}: {obs_dim[role]} vs {agent_obs_dim}")
    
    return role_of, obs_dim