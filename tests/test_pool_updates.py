import pytest
import json
import tempfile
from pathlib import Path
from src.rl.opponent_pool import load_pool, save_pool, register_snapshot, record_match


@pytest.mark.unit
def test_schema_migration():
    """Test automatic migration from schema 1 to schema 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_path = Path(tmpdir) / "test_pool.json"
        
        # Create schema 1 pool manually
        schema_1_pool = {
            "schema": 1,
            "env": "test_env",
            "roles": ["good", "adv"],
            "agents": [
                {
                    "policy_id": "test:good",
                    "role": "good",
                    "ckpt": "/path/to/test.pt",
                    "elo": 1200.0,
                    "games": 5,
                    "wins": 3,
                    "created": "2024-01-01T00:00:00Z"
                }
            ]
        }
        
        # Save schema 1 pool
        with open(pool_path, 'w') as f:
            json.dump(schema_1_pool, f)
        
        # Load should auto-migrate to schema 2
        pool = load_pool(pool_path)
        
        # Should be schema 2 with matches array
        assert pool["schema"] == 2
        assert "matches" in pool
        assert pool["matches"] == []
        
        # Other data should be preserved
        assert pool["env"] == "test_env"
        assert pool["roles"] == ["good", "adv"]
        assert len(pool["agents"]) == 1
        assert pool["agents"][0]["policy_id"] == "test:good"
        
        # File should be auto-saved with schema 2
        with open(pool_path, 'r') as f:
            saved_pool = json.load(f)
        assert saved_pool["schema"] == 2
        assert "matches" in saved_pool


@pytest.mark.unit
def test_new_schema_2_pool():
    """Test creating new schema 2 pool from scratch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_path = Path(tmpdir) / "new_pool.json"
        
        # Load non-existent pool should create schema 2
        pool = load_pool(pool_path)
        
        assert pool["schema"] == 2
        assert pool["env"] == ""
        assert pool["roles"] == []
        assert pool["agents"] == []
        assert pool["matches"] == []


@pytest.mark.unit
def test_record_match_basic():
    """Test basic match recording functionality."""
    pool = {
        "schema": 2,
        "env": "test",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "agent_a",
                "role": "good",
                "ckpt": "/a.pt",
                "elo": 1000.0,
                "games": 0,
                "wins": 0,
                "created": "2024-01-01T00:00:00Z"
            },
            {
                "policy_id": "agent_b", 
                "role": "adv",
                "ckpt": "/b.pt",
                "elo": 1000.0,
                "games": 0,
                "wins": 0,
                "created": "2024-01-01T00:00:00Z"
            }
        ],
        "matches": []
    }
    
    # Record a match where agent_a wins
    record_match(pool, "agent_a", "agent_b", 1.0, "2024-01-02T12:00:00Z")
    
    # Check match was recorded
    assert len(pool["matches"]) == 1
    match = pool["matches"][0]
    assert match["agent_a"] == "agent_a"
    assert match["agent_b"] == "agent_b"
    assert match["score_a"] == 1.0
    assert match["timestamp"] == "2024-01-02T12:00:00Z"
    assert match["elo_a_before"] == 1000.0
    assert match["elo_b_before"] == 1000.0
    
    # Check agents were updated
    agent_a = next(a for a in pool["agents"] if a["policy_id"] == "agent_a")
    agent_b = next(a for a in pool["agents"] if a["policy_id"] == "agent_b")
    
    assert agent_a["games"] == 1
    assert agent_a["wins"] == 1
    assert agent_a["elo"] > 1000.0  # Should increase
    
    assert agent_b["games"] == 1
    assert agent_b["wins"] == 0  # No wins for agent_b
    assert agent_b["elo"] < 1000.0  # Should decrease


@pytest.mark.unit
def test_record_match_draw():
    """Test recording a draw."""
    pool = {
        "schema": 2,
        "env": "test",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "agent_a",
                "role": "good",
                "ckpt": "/a.pt",
                "elo": 1500.0,
                "games": 10,
                "wins": 6,
                "created": "2024-01-01T00:00:00Z"
            },
            {
                "policy_id": "agent_b",
                "role": "adv", 
                "ckpt": "/b.pt",
                "elo": 1300.0,
                "games": 8,
                "wins": 3,
                "created": "2024-01-01T00:00:00Z"
            }
        ],
        "matches": []
    }
    
    # Record a draw
    record_match(pool, "agent_a", "agent_b", 0.5)
    
    # Both agents should have games incremented but no wins
    agent_a = next(a for a in pool["agents"] if a["policy_id"] == "agent_a")
    agent_b = next(a for a in pool["agents"] if a["policy_id"] == "agent_b")
    
    assert agent_a["games"] == 11
    assert agent_a["wins"] == 6  # No additional wins
    assert agent_b["games"] == 9
    assert agent_b["wins"] == 3  # No additional wins


@pytest.mark.unit
def test_record_match_different_k_factors():
    """Test match recording uses appropriate K-factors."""
    pool = {
        "schema": 2,
        "env": "test", 
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "novice",
                "role": "good",
                "ckpt": "/novice.pt", 
                "elo": 1000.0,  # Should get K=40
                "games": 0,
                "wins": 0,
                "created": "2024-01-01T00:00:00Z"
            },
            {
                "policy_id": "expert",
                "role": "adv",
                "ckpt": "/expert.pt",
                "elo": 2500.0,  # Should get K=10  
                "games": 100,
                "wins": 80,
                "created": "2024-01-01T00:00:00Z"
            }
        ],
        "matches": []
    }
    
    initial_novice_elo = 1000.0
    initial_expert_elo = 2500.0
    
    # Novice beats expert (huge upset)
    record_match(pool, "novice", "expert", 1.0)
    
    novice = next(a for a in pool["agents"] if a["policy_id"] == "novice")
    expert = next(a for a in pool["agents"] if a["policy_id"] == "expert")
    
    # Novice should gain significant rating
    novice_gain = novice["elo"] - initial_novice_elo
    expert_loss = initial_expert_elo - expert["elo"]
    
    # Both should change significantly due to upset
    assert novice_gain > 10.0
    assert expert_loss > 10.0
    
    # Rating changes should be equal (conservation)
    assert abs(novice_gain - expert_loss) < 0.01


@pytest.mark.unit
def test_record_match_missing_agent():
    """Test error handling for missing agents."""
    pool = {
        "schema": 2,
        "env": "test",
        "roles": ["good"],
        "agents": [
            {
                "policy_id": "agent_a",
                "role": "good",
                "ckpt": "/a.pt",
                "elo": 1000.0,
                "games": 0,
                "wins": 0,
                "created": "2024-01-01T00:00:00Z"
            }
        ],
        "matches": []
    }
    
    # Should raise error for missing agent
    with pytest.raises(ValueError, match="Agent not found: missing_agent"):
        record_match(pool, "agent_a", "missing_agent", 1.0)
    
    with pytest.raises(ValueError, match="Agent not found: missing_agent"):
        record_match(pool, "missing_agent", "agent_a", 0.0)


@pytest.mark.unit 
def test_record_match_auto_timestamp():
    """Test automatic timestamp generation."""
    pool = {
        "schema": 2,
        "env": "test",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "agent_a",
                "role": "good",
                "ckpt": "/a.pt",
                "elo": 1000.0,
                "games": 0,
                "wins": 0,
                "created": "2024-01-01T00:00:00Z"
            },
            {
                "policy_id": "agent_b",
                "role": "adv", 
                "ckpt": "/b.pt",
                "elo": 1000.0,
                "games": 0,
                "wins": 0,
                "created": "2024-01-01T00:00:00Z"
            }
        ],
        "matches": []
    }
    
    # Record match without timestamp
    record_match(pool, "agent_a", "agent_b", 1.0)
    
    # Should have auto-generated timestamp in ISO8601 format
    match = pool["matches"][0]
    timestamp = match["timestamp"]
    
    # Basic format check (YYYY-MM-DDTHH:MM:SSZ)
    assert len(timestamp) == 20
    assert timestamp[4] == '-'
    assert timestamp[7] == '-'
    assert timestamp[10] == 'T'
    assert timestamp[13] == ':'
    assert timestamp[16] == ':'
    assert timestamp[19] == 'Z'


@pytest.mark.unit
def test_match_history_persistence():
    """Test that match history is preserved across saves/loads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_path = Path(tmpdir) / "history_test.json"
        
        # Create pool with agents and matches
        pool = load_pool(pool_path)
        register_snapshot(pool, "test_env", "good", Path("/a.pt"))
        register_snapshot(pool, "test_env", "adv", Path("/b.pt"))
        
        # Record several matches
        record_match(pool, "a:good", "b:adv", 1.0, "2024-01-01T10:00:00Z")
        record_match(pool, "a:good", "b:adv", 0.0, "2024-01-01T11:00:00Z") 
        record_match(pool, "a:good", "b:adv", 0.5, "2024-01-01T12:00:00Z")
        
        # Save pool
        save_pool(pool_path, pool)
        
        # Load new instance
        loaded_pool = load_pool(pool_path)
        
        # Should have all match history
        assert len(loaded_pool["matches"]) == 3
        assert loaded_pool["matches"][0]["score_a"] == 1.0
        assert loaded_pool["matches"][1]["score_a"] == 0.0
        assert loaded_pool["matches"][2]["score_a"] == 0.5
        
        # Agent stats should be preserved
        agent_good = next(a for a in loaded_pool["agents"] if a["policy_id"] == "a:good")
        agent_adv = next(a for a in loaded_pool["agents"] if a["policy_id"] == "b:adv")
        
        assert agent_good["games"] == 3
        assert agent_good["wins"] == 1  # Won 1 out of 3
        assert agent_adv["games"] == 3
        assert agent_adv["wins"] == 1  # Won 1 out of 3