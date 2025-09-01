"""
Tests for PFSP ramping functionality.
"""

import pytest
from src.rl.ppo_selfplay_skeleton import pfsp_mode_for_epoch


@pytest.mark.unit
def test_pfsp_mode_for_epoch_stages():
    """Test PFSP mode progression through epochs."""
    
    # Stage 1: epochs 0-9 should be uniform
    for epoch in range(0, 10):
        strategy, mode = pfsp_mode_for_epoch(epoch)
        assert strategy == "uniform", f"Epoch {epoch} should be uniform, got {strategy}"
        assert mode is None, f"Epoch {epoch} should have None mode, got {mode}"
    
    # Stage 2: epochs 10-39 should be pfsp/even
    for epoch in range(10, 40):
        strategy, mode = pfsp_mode_for_epoch(epoch)
        assert strategy == "pfsp", f"Epoch {epoch} should be pfsp, got {strategy}"
        assert mode == "even", f"Epoch {epoch} should be even mode, got {mode}"
    
    # Stage 3: epochs 40+ should be pfsp/hard
    for epoch in range(40, 100):  # Test a range of high epochs
        strategy, mode = pfsp_mode_for_epoch(epoch)
        assert strategy == "pfsp", f"Epoch {epoch} should be pfsp, got {strategy}"
        assert mode == "hard", f"Epoch {epoch} should be hard mode, got {mode}"


@pytest.mark.unit
def test_pfsp_mode_boundary_conditions():
    """Test boundary conditions between stages."""
    
    # Just before first transition (epoch 9)
    strategy, mode = pfsp_mode_for_epoch(9)
    assert strategy == "uniform"
    assert mode is None
    
    # Just after first transition (epoch 10)
    strategy, mode = pfsp_mode_for_epoch(10)
    assert strategy == "pfsp"
    assert mode == "even"
    
    # Just before second transition (epoch 39)
    strategy, mode = pfsp_mode_for_epoch(39)
    assert strategy == "pfsp"
    assert mode == "even"
    
    # Just after second transition (epoch 40)
    strategy, mode = pfsp_mode_for_epoch(40)
    assert strategy == "pfsp"
    assert mode == "hard"


@pytest.mark.unit
def test_pfsp_mode_specific_epochs():
    """Test specific epoch values mentioned in requirements."""
    
    # epoch=0 → uniform
    strategy, mode = pfsp_mode_for_epoch(0)
    assert strategy == "uniform"
    assert mode is None
    
    # epoch=20 → pfsp/even  
    strategy, mode = pfsp_mode_for_epoch(20)
    assert strategy == "pfsp"
    assert mode == "even"
    
    # epoch=60 → pfsp/hard
    strategy, mode = pfsp_mode_for_epoch(60)
    assert strategy == "pfsp"
    assert mode == "hard"


@pytest.mark.unit
def test_pfsp_mode_large_epochs():
    """Test behavior with very large epoch values."""
    
    # Should remain pfsp/hard for large epochs
    for epoch in [100, 500, 1000, 10000]:
        strategy, mode = pfsp_mode_for_epoch(epoch)
        assert strategy == "pfsp", f"Large epoch {epoch} should remain pfsp"
        assert mode == "hard", f"Large epoch {epoch} should remain hard mode"


@pytest.mark.unit
def test_pfsp_mode_negative_epochs():
    """Test behavior with negative epoch values (edge case)."""
    
    # Negative epochs should be treated as early epochs (uniform)
    for epoch in [-1, -10, -100]:
        strategy, mode = pfsp_mode_for_epoch(epoch)
        assert strategy == "uniform", f"Negative epoch {epoch} should be uniform"
        assert mode is None, f"Negative epoch {epoch} should have None mode"


@pytest.mark.unit
def test_pfsp_mode_return_types():
    """Test that function returns correct types."""
    
    # Test a few different epochs
    test_epochs = [0, 5, 15, 25, 45, 100]
    
    for epoch in test_epochs:
        strategy, mode = pfsp_mode_for_epoch(epoch)
        
        # Strategy should always be a string
        assert isinstance(strategy, str), f"Strategy should be str, got {type(strategy)}"
        assert strategy in ["uniform", "pfsp"], f"Strategy should be uniform or pfsp, got {strategy}"
        
        # Mode should be string or None
        assert mode is None or isinstance(mode, str), f"Mode should be str or None, got {type(mode)}"
        if mode is not None:
            assert mode in ["hard", "even", "easy"], f"Mode should be hard/even/easy, got {mode}"


@pytest.mark.unit
def test_pfsp_mode_consistency():
    """Test that function returns consistent results for same inputs."""
    
    test_epochs = [0, 9, 10, 39, 40, 100]
    
    for epoch in test_epochs:
        # Call function multiple times
        results = [pfsp_mode_for_epoch(epoch) for _ in range(5)]
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, f"Inconsistent results for epoch {epoch}"


@pytest.mark.unit
def test_pfsp_mode_progression_sequence():
    """Test the complete progression sequence."""
    
    expected_sequence = [
        # Epochs 0-9: uniform
        ("uniform", None),
        # Epochs 10-39: pfsp/even  
        ("pfsp", "even"),
        # Epochs 40+: pfsp/hard
        ("pfsp", "hard")
    ]
    
    # Test progression over 100 epochs
    prev_result = None
    transitions = []
    
    for epoch in range(100):
        result = pfsp_mode_for_epoch(epoch)
        
        if prev_result is not None and prev_result != result:
            transitions.append((epoch, result))
        
        prev_result = result
    
    # Should have exactly 2 transitions (not counting initial state)
    assert len(transitions) == 2, f"Expected 2 transitions, got {len(transitions)}: {transitions}"
    
    # First transition at epoch 10
    assert transitions[0] == (10, ("pfsp", "even")), f"First transition wrong: {transitions[0]}"
    
    # Second transition at epoch 40  
    assert transitions[1] == (40, ("pfsp", "hard")), f"Second transition wrong: {transitions[1]}"


@pytest.mark.unit
def test_pfsp_mode_stage_durations():
    """Test that each stage lasts the expected number of epochs."""
    
    # Stage 1: uniform (epochs 0-9) = 10 epochs
    uniform_count = 0
    for epoch in range(100):
        strategy, mode = pfsp_mode_for_epoch(epoch)
        if strategy == "uniform":
            uniform_count += 1
    assert uniform_count == 10, f"Expected 10 uniform epochs, got {uniform_count}"
    
    # Stage 2: pfsp/even (epochs 10-39) = 30 epochs  
    even_count = 0
    for epoch in range(100):
        strategy, mode = pfsp_mode_for_epoch(epoch)
        if strategy == "pfsp" and mode == "even":
            even_count += 1
    assert even_count == 30, f"Expected 30 even epochs, got {even_count}"
    
    # Stage 3: pfsp/hard (epochs 40-99) = 60 epochs
    hard_count = 0
    for epoch in range(100):
        strategy, mode = pfsp_mode_for_epoch(epoch)
        if strategy == "pfsp" and mode == "hard":
            hard_count += 1  
    assert hard_count == 60, f"Expected 60 hard epochs, got {hard_count}"


@pytest.mark.unit
def test_pfsp_mode_docstring_examples():
    """Test examples from the function docstring."""
    
    # Examples mentioned in comments: 0-9 uniform, 10-39 even, 40+ hard
    
    # Test a few examples from each range
    test_cases = [
        (0, "uniform", None),
        (5, "uniform", None), 
        (9, "uniform", None),
        (10, "pfsp", "even"),
        (20, "pfsp", "even"),
        (39, "pfsp", "even"),
        (40, "pfsp", "hard"),
        (50, "pfsp", "hard"),
        (100, "pfsp", "hard"),
    ]
    
    for epoch, expected_strategy, expected_mode in test_cases:
        strategy, mode = pfsp_mode_for_epoch(epoch)
        assert strategy == expected_strategy, f"Epoch {epoch}: expected strategy {expected_strategy}, got {strategy}"
        assert mode == expected_mode, f"Epoch {epoch}: expected mode {expected_mode}, got {mode}"