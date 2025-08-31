import pytest
import json
import tempfile
from pathlib import Path
from src.rl.opponent_pool import load_pool, save_pool, register_snapshot, by_role


@pytest.mark.unit
def test_load_pool_creates_empty():
    """Test loading pool creates empty pool if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_path = Path(tmpdir) / "test_pool.json"
        
        # Load non-existent pool
        pool = load_pool(pool_path)
        
        # Should create empty pool with correct schema
        assert pool["schema"] == 2
        assert pool["env"] == ""
        assert pool["roles"] == []
        assert pool["agents"] == []
        assert pool["matches"] == []
        
        # File should be created
        assert pool_path.exists()


@pytest.mark.unit
def test_save_load_roundtrip():
    """Test saving and loading pool preserves data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_path = Path(tmpdir) / "test_pool.json"
        
        # Create test pool
        pool = {
            "schema": 2,
            "env": "test_env",
            "roles": ["good", "adv"],
            "agents": [
                {
                    "policy_id": "test1:good",
                    "role": "good",
                    "ckpt": "/path/to/test1.pt",
                    "elo": 1200.0,
                    "games": 10,
                    "wins": 5,
                    "created": "2024-01-01T00:00:00Z"
                }
            ],
            "matches": []
        }
        
        # Save and reload
        save_pool(pool_path, pool)
        loaded = load_pool(pool_path)
        
        # Should match original
        assert loaded == pool


@pytest.mark.unit
def test_register_snapshot_new():
    """Test registering new snapshot."""
    pool = {
        "schema": 2,
        "env": "",
        "roles": [],
        "agents": [],
        "matches": []
    }
    
    ckpt_path = Path("/fake/path/test.pt")
    register_snapshot(pool, "dota_last_hit", "good", ckpt_path, 1100.0)
    
    # Should update pool metadata
    assert pool["env"] == "dota_last_hit"
    assert "good" in pool["roles"]
    
    # Should add agent
    assert len(pool["agents"]) == 1
    agent = pool["agents"][0]
    assert agent["policy_id"] == "test:good"
    assert agent["role"] == "good"
    assert agent["ckpt"] == str(ckpt_path)
    assert agent["elo"] == 1100.0
    assert agent["games"] == 0
    assert agent["wins"] == 0
    assert "created" in agent


@pytest.mark.unit
def test_register_snapshot_duplicate():
    """Test registering duplicate snapshot is no-op."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good"],
        "agents": [
            {
                "policy_id": "test:good",
                "role": "good", 
                "ckpt": "/path/to/test.pt",
                "elo": 1000.0,
                "games": 0,
                "wins": 0,
                "created": "2024-01-01T00:00:00Z"
            }
        ],
        "matches": []
    }
    
    original_count = len(pool["agents"])
    
    # Try to register same snapshot
    ckpt_path = Path("/path/to/test.pt")
    register_snapshot(pool, "test_env", "good", ckpt_path)
    
    # Should not add duplicate
    assert len(pool["agents"]) == original_count


@pytest.mark.unit
def test_register_snapshot_multiple_roles():
    """Test registering multiple roles for same checkpoint."""
    pool = {
        "schema": 2,
        "env": "",
        "roles": [],
        "agents": [],
        "matches": []
    }
    
    ckpt_path = Path("/path/to/multi.pt")
    
    # Register multiple roles
    register_snapshot(pool, "mpe_adversary", "good", ckpt_path)
    register_snapshot(pool, "mpe_adversary", "adv", ckpt_path)
    
    # Should have both roles in metadata
    assert set(pool["roles"]) == {"adv", "good"}  # sorted
    
    # Should have separate agents for each role
    assert len(pool["agents"]) == 2
    agent_roles = {a["role"] for a in pool["agents"]}
    assert agent_roles == {"good", "adv"}
    
    # Policy IDs should be different
    policy_ids = {a["policy_id"] for a in pool["agents"]}
    assert policy_ids == {"multi:good", "multi:adv"}


@pytest.mark.unit
def test_by_role():
    """Test filtering agents by role."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [
            {"policy_id": "a:good", "role": "good", "elo": 1000.0},
            {"policy_id": "b:good", "role": "good", "elo": 1200.0},
            {"policy_id": "c:adv", "role": "adv", "elo": 1100.0}
        ],
        "matches": []
    }
    
    # Filter by role
    good_agents = by_role(pool, "good")
    adv_agents = by_role(pool, "adv")
    empty_agents = by_role(pool, "unknown")
    
    # Should return correct agents
    assert len(good_agents) == 2
    assert len(adv_agents) == 1
    assert len(empty_agents) == 0
    
    # Should be correct agents
    good_ids = {a["policy_id"] for a in good_agents}
    assert good_ids == {"a:good", "b:good"}
    
    adv_ids = {a["policy_id"] for a in adv_agents}
    assert adv_ids == {"c:adv"}


@pytest.mark.unit
def test_atomic_save():
    """Test that save_pool uses atomic writes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_path = Path(tmpdir) / "test_pool.json"
        
        pool = {
            "schema": 2,
            "env": "test",
            "roles": [],
            "agents": [],
            "matches": []
        }
        
        # Save pool
        save_pool(pool_path, pool)
        
        # File should exist and be valid JSON
        assert pool_path.exists()
        with open(pool_path) as f:
            loaded = json.load(f)
        assert loaded == pool
        
        # Temporary file should not exist
        tmp_path = pool_path.with_suffix('.tmp')
        assert not tmp_path.exists()