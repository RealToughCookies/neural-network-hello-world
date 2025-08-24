"""
Unit tests for OpponentPool class.
"""
import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.rl.opponent_pool import OpponentPool


@pytest.fixture
def temp_pool_file():
    """Create temporary file for pool testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def rng():
    """Create seeded RNG for reproducible tests."""
    return np.random.Generator(np.random.PCG64(42))


def test_opponent_pool_creation(temp_pool_file):
    """Test basic pool creation and initialization."""
    pool = OpponentPool(temp_pool_file, max_size=10, ema_decay=0.9, min_games=3)
    
    assert len(pool) == 0
    assert not pool  # Empty pool is falsy
    assert pool.max_size == 10
    assert pool.ema_decay == 0.9
    assert pool.min_games == 3


def test_add_opponents(temp_pool_file):
    """Test adding opponents to pool."""
    pool = OpponentPool(temp_pool_file, max_size=5)
    
    # Add first opponent
    pool.add("ckpt1.pt", step=100)
    assert len(pool) == 1
    assert bool(pool)  # Non-empty pool is truthy
    
    # Add same opponent (should just update step)
    pool.add("ckpt1.pt", step=200)
    assert len(pool) == 1
    
    # Add different opponent
    pool.add("ckpt2.pt", step=150)
    assert len(pool) == 2
    
    # Check persistence
    pool2 = OpponentPool(temp_pool_file, max_size=5)
    assert len(pool2) == 2


def test_record_results(temp_pool_file):
    """Test recording game results and EMA updates."""
    pool = OpponentPool(temp_pool_file, max_size=5, ema_decay=0.8)
    
    pool.add("ckpt1.pt", step=100)
    
    # Record some wins and losses
    pool.record_result("ckpt1.pt", won=True)
    pool.record_result("ckpt1.pt", won=True) 
    pool.record_result("ckpt1.pt", won=False)
    
    # Check opponent state
    opp = pool.opponents[str(Path("ckpt1.pt").resolve())]
    assert opp['wins'] == 2
    assert opp['games'] == 3
    
    # EMA should be between 0.5 and 1.0 (more wins than losses)
    assert 0.5 < opp['ema_wr'] < 1.0


def test_uniform_sampling(temp_pool_file, rng):
    """Test uniform opponent sampling."""
    pool = OpponentPool(temp_pool_file, max_size=5)
    
    # Empty pool
    assert pool.sample_uniform(rng) is None
    
    # Add opponents
    opponents = ["ckpt1.pt", "ckpt2.pt", "ckpt3.pt"]
    for i, ckpt in enumerate(opponents):
        pool.add(ckpt, step=i*100)
    
    # Sample many times and check distribution
    samples = [pool.sample_uniform(rng) for _ in range(300)]
    
    # Each opponent should be sampled roughly equally
    for ckpt in opponents:
        ckpt_resolved = str(Path(ckpt).resolve())
        count = samples.count(ckpt_resolved)
        assert 80 < count < 120  # Should be around 100 ± 20


def test_prioritized_sampling(temp_pool_file, rng):
    """Test prioritized opponent sampling."""
    pool = OpponentPool(temp_pool_file, max_size=5, min_games=2)
    
    # Empty pool
    assert pool.sample_prioritized(rng) is None
    
    # Add opponents with different win rates
    pool.add("weak_ckpt.pt", step=100)
    pool.add("strong_ckpt.pt", step=200)
    
    # Make one opponent weak (low win rate)
    for _ in range(10):
        pool.record_result("weak_ckpt.pt", won=False)
    
    # Make one opponent strong (high win rate)  
    for _ in range(10):
        pool.record_result("strong_ckpt.pt", won=True)
    
    # Sample many times - should favor the stronger opponent
    samples = [pool.sample_prioritized(rng, temp=0.5) for _ in range(200)]
    
    weak_resolved = str(Path("weak_ckpt.pt").resolve())
    strong_resolved = str(Path("strong_ckpt.pt").resolve())
    
    weak_count = samples.count(weak_resolved)
    strong_count = samples.count(strong_resolved)
    
    # Strong opponent should be sampled more often
    assert strong_count > weak_count


def test_pruning(temp_pool_file):
    """Test pool pruning when max size exceeded."""
    pool = OpponentPool(temp_pool_file, max_size=3)
    
    # Add more opponents than max size
    for i in range(5):
        pool.add(f"ckpt{i}.pt", step=i*100)
    
    # Pool should be pruned to max size
    assert len(pool) == 3
    
    # Most recent opponents should be kept (by step number)
    remaining_ckpts = set(pool.opponents.keys())
    for i in [2, 3, 4]:  # Most recent steps
        expected_ckpt = str(Path(f"ckpt{i}.pt").resolve())
        assert expected_ckpt in remaining_ckpts


def test_stats(temp_pool_file):
    """Test pool statistics."""
    pool = OpponentPool(temp_pool_file, max_size=5)
    
    # Empty pool
    stats = pool.get_stats()
    assert stats['size'] == 0
    
    # Add opponents with games
    pool.add("ckpt1.pt", step=100)
    pool.add("ckpt2.pt", step=200)
    
    pool.record_result("ckpt1.pt", won=True)
    pool.record_result("ckpt2.pt", won=False)
    
    stats = pool.get_stats()
    assert stats['size'] == 2
    assert stats['total_games'] == 2
    assert 0 <= stats['avg_ema_wr'] <= 1


def test_decay_all(temp_pool_file):
    """Test global EMA decay toward neutral."""
    pool = OpponentPool(temp_pool_file, max_size=5)
    
    pool.add("ckpt1.pt", step=100)
    
    # Create extreme win rate
    for _ in range(20):
        pool.record_result("ckpt1.pt", won=True)
    
    initial_wr = pool.opponents[str(Path("ckpt1.pt").resolve())]['ema_wr']
    
    # Apply decay
    pool.decay_all(decay_factor=0.9)
    
    final_wr = pool.opponents[str(Path("ckpt1.pt").resolve())]['ema_wr']
    
    # Should move toward 0.5
    assert abs(final_wr - 0.5) < abs(initial_wr - 0.5)


def test_persistence_robustness(temp_pool_file):
    """Test pool handles file corruption and missing files gracefully."""
    # Create pool and add data
    pool1 = OpponentPool(temp_pool_file, max_size=5)
    pool1.add("ckpt1.pt", step=100)
    
    # Corrupt the JSON file
    with open(temp_pool_file, 'w') as f:
        f.write("invalid json {")
    
    # New pool should handle corruption gracefully
    pool2 = OpponentPool(temp_pool_file, max_size=5)
    assert len(pool2) == 0  # Should start fresh
    
    # Should be able to add new opponents
    pool2.add("ckpt2.pt", step=200)
    assert len(pool2) == 1


if __name__ == "__main__":
    pytest.main([__file__])