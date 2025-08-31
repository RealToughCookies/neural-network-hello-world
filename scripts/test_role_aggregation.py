#!/usr/bin/env python3
"""
Test role-based aggregation in the collector.
Verifies that multiple agents per role are handled correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
from src.rl.rollout import collect_rollouts
from src.rl.models import MultiHeadPolicy
from src.rl.env_api import make_adapter

def test_role_aggregation():
    """Test role-based aggregation with both single and multiple agents per role."""
    print("Testing role-based aggregation...")
    
    # Set up logging to see the collector output
    logging.basicConfig(level=logging.INFO)
    
    target_steps = 32
    
    for env_name in ["dota_last_hit", "mpe_adversary"]:
        print(f"\n--- Testing {env_name} ---")
        adapter = make_adapter(env_name)
        
        # Get adapter info
        roles_dict = adapter.roles()
        agents = adapter.agent_names()
        obs_dims = adapter.obs_dims()
        n_actions = adapter.n_actions()
        
        print(f"Agents: {agents}")
        print(f"Roles mapping: {roles_dict}")
        print(f"Unique roles: {sorted(set(roles_dict.values()))}")
        
        # Create policy
        policy = MultiHeadPolicy(obs_dims, n_actions)
        
        # Test collection
        batch, counts = collect_rollouts(
            adapter=adapter,
            policy=policy,
            roles=roles_dict,
            agents=agents,
            per_agent_steps=target_steps,
            seed=42
        )
        
        print(f"Collected counts: {counts}")
        
        # Verify exact step count per role
        unique_roles = sorted(set(roles_dict.values()))
        all_match = all(counts[role] == target_steps for role in unique_roles)
        
        if all_match:
            print(f"✅ {env_name}: All roles collected exactly {target_steps} steps!")
        else:
            print(f"❌ {env_name}: Step counts don't match target!")
            return False
    
    print("\n🎉 All tests passed! Role-based aggregation working correctly.")
    return True

if __name__ == "__main__":
    success = test_role_aggregation()
    sys.exit(0 if success else 1)