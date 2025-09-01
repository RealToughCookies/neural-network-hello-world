"""
Tests for Wilson confidence interval gating logic and checkpoint promotion.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rl.metrics import wilson_ci
from src.rl.ppo_selfplay_skeleton import promote_best_symlink


@pytest.mark.unit
def test_wilson_ci_known_cases():
    """Test Wilson CI with known cases from literature."""
    # Test case: 0 successes out of 10 trials
    lb, ub = wilson_ci(0, 10, alpha=0.05)
    assert lb == 0.0  # Lower bound should be 0
    assert ub < 0.35   # Upper bound should be reasonable for small sample
    
    # Test case: perfect success
    lb, ub = wilson_ci(10, 10, alpha=0.05) 
    assert lb > 0.65   # Lower bound should be high for perfect success
    assert ub == 1.0   # Upper bound should be 1.0
    
    # Test case: empty sample
    lb, ub = wilson_ci(0, 0, alpha=0.05)
    assert lb == 0.0
    assert ub == 1.0
    
    # Test case: 50% success rate with reasonable sample
    lb, ub = wilson_ci(15, 30, alpha=0.05)
    assert 0.30 < lb < 0.50  # Lower bound should be below 50%
    assert 0.50 < ub < 0.70  # Upper bound should be above 50%


@pytest.mark.unit  
def test_wilson_ci_gating_thresholds():
    """Test specific gating scenarios mentioned in requirements."""
    # wins=18, n=30 → LB≈0.42 (no gate)
    lb, ub = wilson_ci(18, 30, alpha=0.05)
    wr_point = 18 / 30
    assert wr_point == 0.6
    assert lb < 0.50  # Should NOT pass 0.50 gate
    assert 0.40 < lb < 0.45  # Should be around 0.42
    
    # wins=21, n=30 → LB≈0.52 (may pass 0.50 gate but not 0.60)
    lb, ub = wilson_ci(21, 30, alpha=0.05)
    wr_point = 21 / 30
    assert wr_point == 0.7
    assert lb >= 0.50  # Should pass 0.50 gate
    assert lb < 0.60   # Should NOT pass 0.60 gate
    assert 0.51 < lb < 0.55  # Should be around 0.52
    
    # wins=24, n=30 → LB≈0.63 (passes both 0.50 and 0.60)
    lb, ub = wilson_ci(24, 30, alpha=0.05)
    wr_point = 24 / 30
    assert wr_point == 0.8
    assert lb >= 0.50  # Should pass 0.50 gate
    assert lb >= 0.60  # Should pass 0.60 gate
    assert 0.62 < lb < 0.65  # Should be around 0.63


@pytest.mark.unit
def test_wilson_ci_confidence_levels():
    """Test different confidence levels."""
    wins, n = 15, 30
    
    # 95% CI (alpha=0.05)
    lb_95, ub_95 = wilson_ci(wins, n, alpha=0.05)
    
    # 90% CI (alpha=0.10) - should be narrower
    lb_90, ub_90 = wilson_ci(wins, n, alpha=0.10)
    
    # 99% CI (alpha=0.01) - should be wider
    lb_99, ub_99 = wilson_ci(wins, n, alpha=0.01)
    
    # Narrower confidence = tighter bounds
    assert lb_90 > lb_95 > lb_99
    assert ub_99 > ub_95 > ub_90


@pytest.mark.unit
def test_promote_best_symlink():
    """Test atomic checkpoint promotion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        
        # Create a mock current checkpoint
        current_ckpt = save_dir / "last.pt"
        current_model = save_dir / "last_model.pt" 
        
        # Write some dummy content
        current_ckpt.write_text("mock checkpoint data")
        current_model.write_text("mock model data")
        
        # Promote to best
        promote_best_symlink(save_dir, current_ckpt)
        
        # Verify best files were created
        best_ckpt = save_dir / "best.pt"
        best_model = save_dir / "best_model.pt"
        
        assert best_ckpt.exists()
        assert best_model.exists()
        assert best_ckpt.read_text() == "mock checkpoint data"
        assert best_model.read_text() == "mock model data"


@pytest.mark.unit
def test_promote_best_symlink_missing_files():
    """Test promotion when source files don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        current_ckpt = save_dir / "nonexistent.pt"
        
        # Should not raise exception
        promote_best_symlink(save_dir, current_ckpt)
        
        # Best files should not be created
        assert not (save_dir / "best.pt").exists()
        assert not (save_dir / "best_model.pt").exists()


@pytest.mark.unit
def test_promote_best_symlink_model_only():
    """Test promotion when only main checkpoint exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir)
        current_ckpt = save_dir / "last.pt"
        
        # Create only main checkpoint, not model version
        current_ckpt.write_text("checkpoint data")
        
        promote_best_symlink(save_dir, current_ckpt)
        
        # Best checkpoint should exist, model version should not
        assert (save_dir / "best.pt").exists()
        assert not (save_dir / "best_model.pt").exists()
        assert (save_dir / "best.pt").read_text() == "checkpoint data"


@pytest.mark.unit  
def test_gating_integration_scenarios():
    """Test complete gating scenarios with different win rates."""
    
    # Scenario 1: Low win rate - no promotion
    wins, games = 18, 30
    lb, ub = wilson_ci(wins, games, alpha=0.05)
    gate_threshold = 0.50
    
    assert lb < gate_threshold  # Should not pass gate
    print(f"Scenario 1: wins={wins}/{games}, wr={wins/games:.3f}, CI=[{lb:.3f}, {ub:.3f}], gate={gate_threshold} → NO PROMOTION")
    
    # Scenario 2: Medium win rate - may pass lower gate
    wins, games = 21, 30  
    lb, ub = wilson_ci(wins, games, alpha=0.05)
    gate_low = 0.50
    gate_high = 0.60
    
    assert lb >= gate_low   # Should pass low gate
    assert lb < gate_high   # Should not pass high gate
    print(f"Scenario 2: wins={wins}/{games}, wr={wins/games:.3f}, CI=[{lb:.3f}, {ub:.3f}], low_gate={gate_low}, high_gate={gate_high} → PROMOTION TO LOW")
    
    # Scenario 3: High win rate - passes both gates
    wins, games = 24, 30
    lb, ub = wilson_ci(wins, games, alpha=0.05) 
    gate_low = 0.50
    gate_high = 0.60
    
    assert lb >= gate_low   # Should pass low gate
    assert lb >= gate_high  # Should pass high gate  
    print(f"Scenario 3: wins={wins}/{games}, wr={wins/games:.3f}, CI=[{lb:.3f}, {ub:.3f}], low_gate={gate_low}, high_gate={gate_high} → PROMOTION TO BOTH")


@pytest.mark.unit
def test_wilson_ci_edge_cases():
    """Test edge cases for Wilson CI."""
    # Single trial success
    lb, ub = wilson_ci(1, 1, alpha=0.05)
    assert 0 < lb < 1
    assert ub == 1.0
    
    # Single trial failure  
    lb, ub = wilson_ci(0, 1, alpha=0.05)
    assert lb == 0.0
    assert 0 < ub < 1
    
    # Large sample with small effect
    lb, ub = wilson_ci(505, 1000, alpha=0.05)  # 50.5% win rate
    wr = 505 / 1000
    # With large sample, CI should be tight around true rate
    assert abs(lb - wr) < 0.05
    assert abs(ub - wr) < 0.05
    
    # Small sample with large effect  
    lb, ub = wilson_ci(9, 10, alpha=0.05)  # 90% win rate
    wr = 9 / 10
    # With small sample, CI should be wide
    assert wr - lb > 0.15  # Lower bound well below point estimate