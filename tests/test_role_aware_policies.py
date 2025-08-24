#!/usr/bin/env python3
"""
Smoke tests for role-aware policies with different observation dimensions.
"""

import sys
import os
import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl.models import MultiHeadPolicy, MultiHeadValue, DimAdapter, PolicyHead, ValueHead
from src.rl.env_utils import get_role_maps


def test_multi_head_policy_forward():
    """Test that MultiHeadPolicy works with different obs dimensions per role."""
    obs_dims = {"good": 10, "adv": 8}
    n_actions = 5
    
    policy = MultiHeadPolicy(obs_dims, n_actions)
    
    # Test forward pass for each role
    good_obs = torch.randn(4, 10)  # Batch of 4 observations
    adv_obs = torch.randn(3, 8)    # Batch of 3 observations
    
    good_logits = policy.forward("good", good_obs)
    adv_logits = policy.forward("adv", adv_obs)
    
    # Check output shapes
    assert good_logits.shape == (4, n_actions), f"Expected (4, {n_actions}), got {good_logits.shape}"
    assert adv_logits.shape == (3, n_actions), f"Expected (3, {n_actions}), got {adv_logits.shape}"
    
    print("✓ MultiHeadPolicy forward passes work correctly")


def test_multi_head_value_forward():
    """Test that MultiHeadValue works with different obs dimensions per role."""
    obs_dims = {"good": 10, "adv": 8}
    
    value_fn = MultiHeadValue(obs_dims)
    
    # Test forward pass for each role
    good_obs = torch.randn(4, 10)
    adv_obs = torch.randn(3, 8)
    
    good_values = value_fn.forward("good", good_obs)
    adv_values = value_fn.forward("adv", adv_obs)
    
    # Check output shapes (should be flattened to 1D)
    assert good_values.shape == (4,), f"Expected (4,), got {good_values.shape}"
    assert adv_values.shape == (3,), f"Expected (3,), got {adv_values.shape}"
    
    print("✓ MultiHeadValue forward passes work correctly")


def test_dimension_adapter():
    """Test that DimAdapter correctly transforms observation dimensions."""
    adapter = DimAdapter(in_dim=8, out_dim=10)
    
    # Test transformation
    small_obs = torch.randn(5, 8)
    large_obs = adapter(small_obs)
    
    assert large_obs.shape == (5, 10), f"Expected (5, 10), got {large_obs.shape}"
    
    print("✓ DimAdapter works correctly")


def test_legacy_policy_compatibility():
    """Test that legacy PolicyHead and ValueHead still work."""
    policy = PolicyHead(in_dim=10, n_act=5)
    value_fn = ValueHead(in_dim=10)
    
    obs = torch.randn(3, 10)
    
    logits = policy(obs)
    values = value_fn(obs)
    
    assert logits.shape == (3, 5), f"Expected (3, 5), got {logits.shape}"
    assert values.shape == (3,), f"Expected (3,), got {values.shape}"
    
    print("✓ Legacy PolicyHead and ValueHead work correctly")


def test_role_head_access():
    """Test that we can access individual role heads."""
    obs_dims = {"good": 10, "adv": 8}
    policy = MultiHeadPolicy(obs_dims, n_actions=5)
    value_fn = MultiHeadValue(obs_dims)
    
    # Get individual heads
    good_policy_head = policy.get_role_head("good")
    adv_policy_head = policy.get_role_head("adv")
    good_value_head = value_fn.get_role_head("good")
    adv_value_head = value_fn.get_role_head("adv")
    
    # Test they work independently
    good_obs = torch.randn(2, 10)
    adv_obs = torch.randn(2, 8)
    
    good_logits = good_policy_head(good_obs)
    adv_logits = adv_policy_head(adv_obs)
    good_values = good_value_head(good_obs).squeeze(-1)
    adv_values = adv_value_head(adv_obs).squeeze(-1)
    
    assert good_logits.shape == (2, 5)
    assert adv_logits.shape == (2, 5)
    assert good_values.shape == (2,)
    assert adv_values.shape == (2,)
    
    print("✓ Role head access works correctly")


def test_invalid_role_handling():
    """Test that invalid roles raise appropriate errors."""
    obs_dims = {"good": 10, "adv": 8}
    policy = MultiHeadPolicy(obs_dims, n_actions=5)
    
    try:
        obs = torch.randn(1, 10)
        policy.forward("invalid_role", obs)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown role" in str(e)
        print("✓ Invalid role handling works correctly")


def run_all_tests():
    """Run all smoke tests for role-aware policies."""
    print("Running role-aware policy smoke tests...")
    
    test_multi_head_policy_forward()
    test_multi_head_value_forward()
    test_dimension_adapter()
    test_legacy_policy_compatibility()
    test_role_head_access()
    test_invalid_role_handling()
    
    print("\n✅ All role-aware policy tests passed!")


if __name__ == "__main__":
    run_all_tests()