"""
Tests for opponent pool pruning functionality.
"""

import pytest
from src.rl.opponent_pool import prune_pool_by_created


@pytest.mark.unit
def test_prune_pool_basic():
    """Test basic pool pruning functionality."""
    # Create pool with 60 agents
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [],
        "matches": []
    }
    
    # Add 60 agents with incremental timestamps
    for i in range(60):
        agent = {
            "policy_id": f"agent_{i:03d}:good",
            "role": "good",
            "ckpt": f"/path/to/agent_{i:03d}.pt",
            "elo": 1000.0 + i,  # Incrementing Elo
            "games": 10,
            "wins": 5,
            "created": f"2024-01-01T{i//10:02d}:{i%10*6:02d}:00Z"  # Incrementing timestamps
        }
        pool["agents"].append(agent)
    
    # Verify we have 60 agents
    assert len(pool["agents"]) == 60
    
    # Prune to 50 agents
    prune_pool_by_created(pool, cap=50)
    
    # Should have exactly 50 agents
    assert len(pool["agents"]) == 50
    
    # Should keep the 50 most recent (highest indices)
    remaining_indices = [int(agent["policy_id"].split("_")[1].split(":")[0]) for agent in pool["agents"]]
    expected_indices = list(range(10, 60))  # Agents 10-59 (50 agents)
    
    assert remaining_indices == expected_indices


@pytest.mark.unit
def test_prune_pool_keep_most_recent():
    """Test that pruning keeps the most recent agents by timestamp."""
    pool = {
        "schema": 2,
        "env": "test_env", 
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "old_agent:good",
                "role": "good",
                "ckpt": "/old.pt",
                "elo": 1000.0,
                "games": 10,
                "wins": 5,
                "created": "2024-01-01T00:00:00Z"  # Oldest
            },
            {
                "policy_id": "middle_agent:good", 
                "role": "good",
                "ckpt": "/middle.pt",
                "elo": 1100.0,
                "games": 10,
                "wins": 6,
                "created": "2024-01-01T12:00:00Z"  # Middle
            },
            {
                "policy_id": "new_agent:good",
                "role": "good", 
                "ckpt": "/new.pt",
                "elo": 1200.0,
                "games": 10,
                "wins": 7,
                "created": "2024-01-01T23:59:59Z"  # Newest
            }
        ],
        "matches": []
    }
    
    # Prune to 2 agents
    prune_pool_by_created(pool, cap=2)
    
    # Should keep the 2 most recent
    assert len(pool["agents"]) == 2
    policy_ids = [agent["policy_id"] for agent in pool["agents"]]
    assert "old_agent:good" not in policy_ids  # Oldest should be removed
    assert "middle_agent:good" in policy_ids
    assert "new_agent:good" in policy_ids


@pytest.mark.unit  
def test_prune_pool_no_created_field():
    """Test pruning behavior when agents lack 'created' field."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "agent_1:good",
                "role": "good",
                "ckpt": "/agent1.pt",
                "elo": 1000.0,
                # No 'created' field - should sort to beginning
            },
            {
                "policy_id": "agent_2:good",
                "role": "good", 
                "ckpt": "/agent2.pt",
                "elo": 1100.0,
                "created": "2024-01-01T12:00:00Z"
            },
            {
                "policy_id": "agent_3:good",
                "role": "good",
                "ckpt": "/agent3.pt", 
                "elo": 1200.0,
                # No 'created' field
            }
        ],
        "matches": []
    }
    
    # Prune to 2 agents
    prune_pool_by_created(pool, cap=2)
    
    # Should keep 2 agents - the one with timestamp and one without
    assert len(pool["agents"]) == 2
    
    # Agent with timestamp should definitely be kept
    policy_ids = [agent["policy_id"] for agent in pool["agents"]]
    assert "agent_2:good" in policy_ids


@pytest.mark.unit
def test_prune_pool_already_under_cap():
    """Test that pruning does nothing when pool is already under capacity."""
    original_pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "agent_1:good",
                "role": "good",
                "ckpt": "/agent1.pt", 
                "elo": 1000.0,
                "games": 10,
                "wins": 5,
                "created": "2024-01-01T00:00:00Z"
            },
            {
                "policy_id": "agent_2:good",
                "role": "good",
                "ckpt": "/agent2.pt",
                "elo": 1100.0, 
                "games": 10,
                "wins": 6,
                "created": "2024-01-01T12:00:00Z"
            }
        ],
        "matches": []
    }
    
    # Make a deep copy to compare
    import copy
    pool = copy.deepcopy(original_pool)
    
    # Prune to 10 agents (more than we have)
    prune_pool_by_created(pool, cap=10)
    
    # Pool should be unchanged
    assert pool == original_pool
    assert len(pool["agents"]) == 2


@pytest.mark.unit
def test_prune_pool_empty():
    """Test pruning an empty pool."""
    pool = {
        "schema": 2,
        "env": "test_env", 
        "roles": ["good", "adv"],
        "agents": [],
        "matches": []
    }
    
    # Should not crash
    prune_pool_by_created(pool, cap=10)
    
    # Should remain empty
    assert len(pool["agents"]) == 0


@pytest.mark.unit
def test_prune_pool_exact_cap():
    """Test pruning when pool size equals capacity."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"], 
        "agents": [
            {
                "policy_id": "agent_1:good",
                "role": "good",
                "ckpt": "/agent1.pt",
                "elo": 1000.0,
                "created": "2024-01-01T00:00:00Z"
            },
            {
                "policy_id": "agent_2:good", 
                "role": "good",
                "ckpt": "/agent2.pt",
                "elo": 1100.0,
                "created": "2024-01-01T12:00:00Z"  
            }
        ],
        "matches": []
    }
    
    import copy
    original_pool = copy.deepcopy(pool)
    
    # Prune to exact current size
    prune_pool_by_created(pool, cap=2)
    
    # Should be unchanged
    assert pool == original_pool


@pytest.mark.unit
def test_prune_pool_mixed_roles():
    """Test pruning with agents of different roles."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "good_1:good",
                "role": "good",
                "ckpt": "/good1.pt", 
                "elo": 1000.0,
                "created": "2024-01-01T01:00:00Z"
            },
            {
                "policy_id": "adv_1:adv",
                "role": "adv",
                "ckpt": "/adv1.pt",
                "elo": 1050.0,
                "created": "2024-01-01T02:00:00Z"
            },
            {
                "policy_id": "good_2:good",
                "role": "good", 
                "ckpt": "/good2.pt",
                "elo": 1100.0,
                "created": "2024-01-01T03:00:00Z"
            },
            {
                "policy_id": "adv_2:adv",
                "role": "adv",
                "ckpt": "/adv2.pt",
                "elo": 1150.0,
                "created": "2024-01-01T04:00:00Z"  # Most recent
            }
        ],
        "matches": []
    }
    
    # Prune to 2 agents
    prune_pool_by_created(pool, cap=2)
    
    # Should keep the 2 most recent regardless of role
    assert len(pool["agents"]) == 2
    policy_ids = [agent["policy_id"] for agent in pool["agents"]]
    
    # Should keep good_2 and adv_2 (most recent)
    assert "good_2:good" in policy_ids
    assert "adv_2:adv" in policy_ids
    assert "good_1:good" not in policy_ids  # Oldest - removed
    assert "adv_1:adv" not in policy_ids   # Second oldest - removed


@pytest.mark.unit
def test_prune_pool_large_scale():
    """Test pruning with a large number of agents to verify performance.""" 
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [],
        "matches": []
    }
    
    # Create 1000 agents with varying timestamps
    import random
    random.seed(42)  # For reproducible test
    
    for i in range(1000):
        # Create timestamps with some randomness but generally increasing
        base_time = i * 100  # Base increment
        noise = random.randint(-50, 50)  # Add some noise
        timestamp_seconds = max(0, base_time + noise)
        
        agent = {
            "policy_id": f"agent_{i:04d}:good",
            "role": "good",
            "ckpt": f"/agent_{i:04d}.pt",
            "elo": 1000.0 + random.random() * 500,
            "created": f"2024-01-01T{(timestamp_seconds // 3600) % 24:02d}:{(timestamp_seconds // 60) % 60:02d}:{timestamp_seconds % 60:02d}Z"
        }
        pool["agents"].append(agent)
    
    # Prune to 100 agents
    prune_pool_by_created(pool, cap=100)
    
    # Should have exactly 100 agents
    assert len(pool["agents"]) == 100
    
    # Verify they are sorted by created timestamp
    timestamps = [agent["created"] for agent in pool["agents"]]
    assert timestamps == sorted(timestamps)