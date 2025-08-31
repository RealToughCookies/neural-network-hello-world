import pytest
import numpy as np
from src.rl.opponent_pool import sample_uniform, sample_topk, sample_pfsp_elo


def create_test_pool():
    """Create test pool with varied Elo ratings."""
    return {
        "schema": 1,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [
            {"policy_id": "weak:good", "role": "good", "elo": 800.0},
            {"policy_id": "avg1:good", "role": "good", "elo": 1000.0},
            {"policy_id": "avg2:good", "role": "good", "elo": 1000.0},
            {"policy_id": "strong1:good", "role": "good", "elo": 1200.0},
            {"policy_id": "strong2:good", "role": "good", "elo": 1300.0},
            {"policy_id": "adv1:adv", "role": "adv", "elo": 900.0},
            {"policy_id": "adv2:adv", "role": "adv", "elo": 1100.0}
        ]
    }


@pytest.mark.unit
def test_sample_uniform_seeded():
    """Test uniform sampling with seeded RNG gives reproducible results."""
    pool = create_test_pool()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    # Same seed should give same results
    samples1 = sample_uniform(pool, "good", 10, rng1)
    samples2 = sample_uniform(pool, "good", 10, rng2)
    
    assert len(samples1) == 10
    assert len(samples2) == 10
    
    # Should be identical
    policy_ids1 = [a["policy_id"] for a in samples1]
    policy_ids2 = [a["policy_id"] for a in samples2]
    assert policy_ids1 == policy_ids2


@pytest.mark.unit
def test_sample_uniform_distribution():
    """Test uniform sampling approximates uniform distribution."""
    pool = create_test_pool()
    rng = np.random.default_rng(123)
    
    # Sample many times
    samples = sample_uniform(pool, "good", 1000, rng)
    
    # Count each agent
    counts = {}
    for agent in samples:
        pid = agent["policy_id"]
        counts[pid] = counts.get(pid, 0) + 1
    
    # Should have roughly equal counts (5 good agents, ~200 each)
    expected = 1000 / 5  # 200
    for count in counts.values():
        assert abs(count - expected) < 50  # Allow ±50 variance


@pytest.mark.unit 
def test_sample_topk():
    """Test top-k sampling only returns top agents."""
    pool = create_test_pool()
    rng = np.random.default_rng(456)
    
    # Sample top-2 agents from good role
    samples = sample_topk(pool, "good", 2, 100, rng)
    
    assert len(samples) == 100
    
    # Should only contain top-2 by Elo (strong1:1200, strong2:1300)
    policy_ids = {a["policy_id"] for a in samples}
    assert policy_ids == {"strong1:good", "strong2:good"}
    
    # Count distribution
    counts = {}
    for agent in samples:
        pid = agent["policy_id"]
        counts[pid] = counts.get(pid, 0) + 1
    
    # Both should be sampled (roughly equal since uniform within top-k)
    assert "strong1:good" in counts
    assert "strong2:good" in counts
    assert counts["strong1:good"] + counts["strong2:good"] == 100


@pytest.mark.unit
def test_sample_topk_k_larger_than_pool():
    """Test top-k when k is larger than available agents."""
    pool = create_test_pool()
    rng = np.random.default_rng(789)
    
    # Request top-10 but only 5 good agents exist
    samples = sample_topk(pool, "good", 10, 50, rng)
    
    assert len(samples) == 50
    
    # Should contain all 5 good agents
    policy_ids = {a["policy_id"] for a in samples}
    expected_ids = {"weak:good", "avg1:good", "avg2:good", "strong1:good", "strong2:good"}
    assert policy_ids == expected_ids


@pytest.mark.unit
def test_sample_pfsp_elo_hard_mode():
    """Test PFSP sampling in hard mode prefers strong opponents."""
    pool = create_test_pool()
    rng = np.random.default_rng(999)
    agent_elo = 1000.0  # Mid-level agent
    
    # Sample with hard mode (prefer harder opponents)
    samples = sample_pfsp_elo(pool, "good", agent_elo, "hard", 2.0, 500, rng)
    
    assert len(samples) == 500
    
    # Count Elo distribution
    elo_counts = {}
    for agent in samples:
        elo = agent["elo"]
        elo_counts[elo] = elo_counts.get(elo, 0) + 1
    
    # Higher Elo agents should be sampled more often
    strong_count = elo_counts.get(1200.0, 0) + elo_counts.get(1300.0, 0)  # strong agents
    weak_count = elo_counts.get(800.0, 0)  # weak agent
    
    assert strong_count > weak_count


@pytest.mark.unit
def test_sample_pfsp_elo_easy_mode():
    """Test PFSP sampling in easy mode prefers weak opponents."""
    pool = create_test_pool()
    rng = np.random.default_rng(111)
    agent_elo = 1000.0  # Mid-level agent
    
    # Sample with easy mode (prefer easier opponents)
    samples = sample_pfsp_elo(pool, "good", agent_elo, "easy", 2.0, 500, rng)
    
    assert len(samples) == 500
    
    # Count Elo distribution
    elo_counts = {}
    for agent in samples:
        elo = agent["elo"]
        elo_counts[elo] = elo_counts.get(elo, 0) + 1
    
    # Lower Elo agents should be sampled more often
    strong_count = elo_counts.get(1200.0, 0) + elo_counts.get(1300.0, 0)  # strong agents
    weak_count = elo_counts.get(800.0, 0)  # weak agent
    
    assert weak_count > strong_count / 2  # Weak should be more common than either strong agent


@pytest.mark.unit
def test_sample_pfsp_elo_even_mode():
    """Test PFSP sampling in even mode prefers balanced opponents."""
    pool = create_test_pool()
    rng = np.random.default_rng(222)
    agent_elo = 1000.0  # Mid-level agent
    
    # Sample with even mode (prefer balanced matchups)
    samples = sample_pfsp_elo(pool, "good", agent_elo, "even", 2.0, 500, rng)
    
    assert len(samples) == 500
    
    # Count Elo distribution
    elo_counts = {}
    for agent in samples:
        elo = agent["elo"]
        elo_counts[elo] = elo_counts.get(elo, 0) + 1
    
    # Mid-level agents (1000) should be most common
    avg_count = elo_counts.get(1000.0, 0)  # 2 agents at 1000 Elo
    weak_count = elo_counts.get(800.0, 0)   # 1 agent at 800
    strong_count = elo_counts.get(1200.0, 0) + elo_counts.get(1300.0, 0)  # 2 agents at high Elo
    
    # Average Elo agents should be sampled most (closest to balanced)
    assert avg_count > weak_count
    assert avg_count > strong_count / 2  # More than any individual strong agent


@pytest.mark.unit
def test_sample_pfsp_elo_different_powers():
    """Test PFSP sampling with different power parameters."""
    pool = create_test_pool()
    agent_elo = 1000.0
    
    rng1 = np.random.default_rng(333)
    rng2 = np.random.default_rng(333)
    
    # Low power = less focused weighting
    samples_p1 = sample_pfsp_elo(pool, "good", agent_elo, "hard", 1.0, 300, rng1)
    
    # High power = more focused weighting  
    samples_p4 = sample_pfsp_elo(pool, "good", agent_elo, "hard", 4.0, 300, rng2)
    
    # Count strong agents in each
    def count_strong(samples):
        return sum(1 for a in samples if a["elo"] >= 1200.0)
    
    strong_p1 = count_strong(samples_p1)
    strong_p4 = count_strong(samples_p4)
    
    # Higher power should focus more on strong opponents
    assert strong_p4 > strong_p1


@pytest.mark.unit
def test_sample_empty_role():
    """Test sampling from non-existent role returns empty list."""
    pool = create_test_pool()
    rng = np.random.default_rng(444)
    
    # Sample from role that doesn't exist
    samples = sample_uniform(pool, "unknown", 10, rng)
    assert samples == []
    
    samples = sample_topk(pool, "unknown", 5, 10, rng)
    assert samples == []
    
    samples = sample_pfsp_elo(pool, "unknown", 1000.0, "even", 2.0, 10, rng)
    assert samples == []


@pytest.mark.unit
def test_sample_empty_pool():
    """Test sampling from empty pool returns empty list."""
    empty_pool = {
        "schema": 1,
        "env": "test",
        "roles": [],
        "agents": []
    }
    rng = np.random.default_rng(555)
    
    samples = sample_uniform(empty_pool, "good", 10, rng)
    assert samples == []
    
    samples = sample_topk(empty_pool, "good", 5, 10, rng)
    assert samples == []
    
    samples = sample_pfsp_elo(empty_pool, "good", 1000.0, "even", 2.0, 10, rng)
    assert samples == []