"""
Integration test for gating system - verifies all components work together.
"""

import pytest
import tempfile
from pathlib import Path

from src.rl.metrics import wilson_ci
from src.rl.opponent_pool import prune_pool_by_created, load_pool, save_pool
from src.rl.ppo_selfplay_skeleton import pfsp_mode_for_epoch, promote_best_symlink


@pytest.mark.unit
def test_end_to_end_gating_workflow():
    """Test complete gating workflow with Wilson CI."""
    
    # Test Wilson CI with various scenarios
    test_cases = [
        (18, 30, 0.50, False),  # Should NOT pass 0.50 gate
        (21, 30, 0.50, True),   # Should pass 0.50 gate
        (21, 30, 0.60, False),  # Should NOT pass 0.60 gate  
        (24, 30, 0.60, True),   # Should pass 0.60 gate
    ]
    
    for wins, games, threshold, should_pass in test_cases:
        lb, ub = wilson_ci(wins, games, alpha=0.05)
        passes_gate = lb >= threshold
        
        print(f"wins={wins}, games={games}, LB={lb:.3f}, gate={threshold}, passes={passes_gate}")
        assert passes_gate == should_pass, f"Gate logic failed for {wins}/{games} vs {threshold}"


@pytest.mark.unit 
def test_pool_lifecycle_with_gating():
    """Test pool creation, population, pruning cycle."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pool_file = Path(tmpdir) / "test_pool.json"
        
        # Create empty pool
        pool = {
            "schema": 2,
            "env": "test_env",
            "roles": ["good", "adv"],
            "agents": [],
            "matches": []
        }
        
        # Add agents with different timestamps
        for i in range(10):
            agent = {
                "policy_id": f"agent_{i:02d}:good",
                "role": "good",
                "ckpt": f"/path/to/agent_{i:02d}.pt",
                "elo": 1000.0 + i * 10,
                "games": 5,
                "wins": 3,
                "created": f"2024-01-{i+1:02d}T12:00:00Z"
            }
            pool["agents"].append(agent)
        
        # Save pool
        save_pool(pool_file, pool)
        
        # Load pool back
        loaded_pool = load_pool(pool_file)
        assert len(loaded_pool["agents"]) == 10
        
        # Prune to smaller size
        prune_pool_by_created(loaded_pool, cap=5)
        assert len(loaded_pool["agents"]) == 5
        
        # Verify we kept the most recent
        remaining_ids = [agent["policy_id"] for agent in loaded_pool["agents"]]
        expected_ids = [f"agent_{i:02d}:good" for i in range(5, 10)]  # agents 05-09
        assert remaining_ids == expected_ids


@pytest.mark.unit
def test_pfsp_ramping_integration():
    """Test PFSP ramping integrates correctly."""
    
    # Test epoch progression
    results = {}
    for epoch in range(0, 100, 10):
        strategy, mode = pfsp_mode_for_epoch(epoch) 
        results[epoch] = (strategy, mode)
    
    # Verify expected progression
    assert results[0] == ("uniform", None)
    assert results[10] == ("pfsp", "even")  
    assert results[20] == ("pfsp", "even")
    assert results[30] == ("pfsp", "even")
    assert results[40] == ("pfsp", "hard")
    assert results[50] == ("pfsp", "hard")
    
    # Test specific boundary conditions
    assert pfsp_mode_for_epoch(9) == ("uniform", None)
    assert pfsp_mode_for_epoch(10) == ("pfsp", "even")
    assert pfsp_mode_for_epoch(39) == ("pfsp", "even") 
    assert pfsp_mode_for_epoch(40) == ("pfsp", "hard")


@pytest.mark.unit
def test_checkpoint_promotion():
    """Test best checkpoint promotion."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        
        # Create mock current checkpoint
        current_ckpt = save_dir / "last.pt"
        current_model = save_dir / "last_model.pt"
        
        current_ckpt.write_text("mock checkpoint data")
        current_model.write_text("mock model data")
        
        # Test promotion
        promote_best_symlink(save_dir, current_ckpt)
        
        # Verify files were created
        best_ckpt = save_dir / "best.pt"
        best_model = save_dir / "best_model.pt"
        
        assert best_ckpt.exists()
        assert best_model.exists()
        assert best_ckpt.read_text() == "mock checkpoint data"
        assert best_model.read_text() == "mock model data"


@pytest.mark.unit 
def test_gating_thresholds_realistic():
    """Test with realistic win/loss scenarios."""
    
    scenarios = [
        # Low performance - no gates should pass (LB=0.246)
        {"wins": 12, "games": 30, "gates": [0.20, 0.30], "expected": [True, False]},
        
        # Medium performance - lower gate passes (LB=0.392)
        {"wins": 17, "games": 30, "gates": [0.35, 0.45], "expected": [True, False]},
        
        # High performance - both gates pass (LB=0.556) 
        {"wins": 22, "games": 30, "gates": [0.50, 0.55], "expected": [True, True]},
        
        # Small sample edge case - too uncertain (LB=0.231)
        {"wins": 3, "games": 5, "gates": [0.20, 0.30], "expected": [True, False]},
        
        # Large sample precision - clear signal (LB=0.423)
        {"wins": 52, "games": 100, "gates": [0.40, 0.45], "expected": [True, False]},
    ]
    
    for scenario in scenarios:
        wins = scenario["wins"]
        games = scenario["games"] 
        gates = scenario["gates"]
        expected = scenario["expected"]
        
        lb, ub = wilson_ci(wins, games, alpha=0.05)
        
        for i, gate_threshold in enumerate(gates):
            passes_gate = lb >= gate_threshold
            should_pass = expected[i]
            
            print(f"Scenario: {wins}/{games} vs gate {gate_threshold}: LB={lb:.3f}, passes={passes_gate}")
            assert passes_gate == should_pass, f"Gate {gate_threshold} failed for {wins}/{games}"


if __name__ == "__main__":
    # Run tests directly
    test_end_to_end_gating_workflow()
    test_pool_lifecycle_with_gating()
    test_pfsp_ramping_integration() 
    test_checkpoint_promotion()
    test_gating_thresholds_realistic()
    print("✅ All integration tests passed!")