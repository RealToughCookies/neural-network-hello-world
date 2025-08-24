#!/usr/bin/env python3
"""
Quick smoke test for environment adapter functionality.
"""

def test_adapter():
    """Test that the adapter works correctly and has expected properties."""
    from src.rl.env_api import make_adapter
    import src.rl.adapters  # Register adapters
    
    # Create adapter
    adapter = make_adapter("mpe_adversary", render_mode=None)
    
    # Test basic functionality
    print("=== Adapter Smoke Test ===")
    
    # Test properties
    roles = adapter.roles()
    obs_dims = adapter.obs_dims()
    n_actions = adapter.n_actions()
    agent_names = adapter.agent_names()
    
    print(f"Roles: {roles}")
    print(f"Obs dims: {obs_dims}")
    print(f"N actions: {n_actions}")
    print(f"Agent names: {agent_names}")
    
    # Verify expected properties
    assert "good" in obs_dims, "Should have 'good' role dimension"
    assert "adv" in obs_dims, "Should have 'adv' role dimension"
    assert obs_dims["good"] != obs_dims["adv"], f"Dimensions should be different: good={obs_dims['good']}, adv={obs_dims['adv']}"
    assert n_actions == 5, f"Expected 5 actions, got {n_actions}"
    assert len(agent_names) > 0, "Should have at least one agent"
    
    # Test typical dimensions (good=10, adv=8 for simple_adversary)
    print(f"✓ good_dim={obs_dims['good']}, adv_dim={obs_dims['adv']}")
    print(f"✓ n_actions={n_actions}")
    
    # Test reset
    ts = adapter.reset(seed=42)
    print(f"✓ Reset successful, obs keys: {list(ts.obs.keys())}")
    print(f"✓ Obs shapes: {[(k, v.shape) for k, v in ts.obs.items()]}")
    
    # Test step
    actions = {name: 0 for name in agent_names}  # No-op actions
    ts = adapter.step(actions)
    print(f"✓ Step successful, rewards: {ts.rewards}")
    
    # Clean up
    adapter.close()
    
    print("✓ All adapter tests passed!")

if __name__ == "__main__":
    test_adapter()