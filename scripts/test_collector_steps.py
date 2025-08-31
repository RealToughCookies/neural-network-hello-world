#!/usr/bin/env python3
"""
Simple smoke test for agent-step counting in the collector.
Verifies that exactly N steps are collected per role.
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

def test_agent_step_counting():
    """Test that collector produces exactly N steps per role."""
    print("Testing agent-step counting with --rollout-steps 64...")
    
    # Set up logging to see the collector output
    logging.basicConfig(level=logging.INFO)
    
    # Create a simple environment
    adapter = make_adapter("dota_last_hit")
    
    # Get adapter info
    roles_dict = adapter.roles()
    agents = adapter.agent_names()
    obs_dims = adapter.obs_dims()
    n_actions = adapter.n_actions()
    
    print(f"Environment: {len(agents)} agents, roles: {roles_dict}")
    
    # Create policy
    policy = MultiHeadPolicy(obs_dims, n_actions)
    
    # Test collection with 64 steps
    target_steps = 64
    batch, counts = collect_rollouts(
        adapter=adapter,
        policy=policy,
        roles=roles_dict,
        agents=agents,
        per_agent_steps=target_steps,
        seed=42
    )
    
    print(f"Collected steps: {counts}")
    print(f"Target steps: {target_steps}")
    
    # Verify exact step count
    all_match = all(counts[role] == target_steps for role in counts)
    if all_match:
        print("✅ SUCCESS: All roles collected exactly the target number of steps!")
        return True
    else:
        print("❌ FAIL: Step counts don't match target!")
        return False

if __name__ == "__main__":
    success = test_agent_step_counting()
    sys.exit(0 if success else 1)