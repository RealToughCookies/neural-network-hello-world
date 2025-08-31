import pytest
import math
from src.rl.elo import expected_score, update, k_factor


@pytest.mark.unit
def test_expected_score_equal():
    """Test expected score for equal ratings."""
    result = expected_score(1600, 1600)
    assert abs(result - 0.5) < 1e-6


@pytest.mark.unit 
def test_expected_score_advantage():
    """Test expected score for higher rated player."""
    result = expected_score(1800, 1600)
    expected = 1.0 / (1.0 + 10 ** ((1600 - 1800) / 400.0))
    assert abs(result - expected) < 1e-6
    # From Elo formula: ~0.759746
    assert abs(result - 0.759746) < 1e-5


@pytest.mark.unit
def test_expected_score_disadvantage():
    """Test expected score for lower rated player.""" 
    result = expected_score(1600, 1800)
    expected = 1.0 / (1.0 + 10 ** ((1800 - 1600) / 400.0))
    assert abs(result - expected) < 1e-6
    # Should be complement of advantage test
    assert abs(result - 0.240254) < 1e-5


@pytest.mark.unit
def test_update_winner():
    """Test Elo update when A wins."""
    ra_old, rb_old = 1600.0, 1600.0
    ra_new, rb_new = update(ra_old, rb_old, score_a=1.0, k=32.0)
    
    # A should gain rating, B should lose rating
    assert ra_new > ra_old
    assert rb_new < rb_old
    
    # Total rating should be conserved
    assert abs((ra_new + rb_new) - (ra_old + rb_old)) < 1e-10


@pytest.mark.unit
def test_update_loser():
    """Test Elo update when A loses."""
    ra_old, rb_old = 1600.0, 1600.0
    ra_new, rb_new = update(ra_old, rb_old, score_a=0.0, k=32.0)
    
    # A should lose rating, B should gain rating
    assert ra_new < ra_old
    assert rb_new > rb_old
    
    # Total rating should be conserved
    assert abs((ra_new + rb_new) - (ra_old + rb_old)) < 1e-10


@pytest.mark.unit
def test_update_draw():
    """Test Elo update for draw."""
    ra_old, rb_old = 1600.0, 1600.0
    ra_new, rb_new = update(ra_old, rb_old, score_a=0.5, k=32.0)
    
    # Ratings should stay the same for equal players in draw
    assert abs(ra_new - ra_old) < 1e-10
    assert abs(rb_new - rb_old) < 1e-10


@pytest.mark.unit
def test_update_asymmetric():
    """Test Elo update with different starting ratings."""
    ra_old, rb_old = 1800.0, 1600.0
    
    # Higher rated player wins (expected outcome)
    ra_new1, rb_new1 = update(ra_old, rb_old, score_a=1.0, k=32.0)
    # Rating change should be smaller for expected outcome
    rating_change_expected = abs(ra_new1 - ra_old)
    
    # Higher rated player loses (upset)
    ra_new2, rb_new2 = update(ra_old, rb_old, score_a=0.0, k=32.0)
    # Rating change should be larger for upset
    rating_change_upset = abs(ra_new2 - ra_old)
    
    assert rating_change_upset > rating_change_expected


@pytest.mark.unit
def test_update_different_k():
    """Test Elo update with different K factors."""
    ra_old, rb_old = 1600.0, 1600.0
    
    ra_k16, rb_k16 = update(ra_old, rb_old, score_a=1.0, k=16.0)
    ra_k32, rb_k32 = update(ra_old, rb_old, score_a=1.0, k=32.0)
    
    # Higher K should produce larger rating changes
    change_k16 = abs(ra_k16 - ra_old)
    change_k32 = abs(ra_k32 - ra_old)
    
    assert change_k32 > change_k16


@pytest.mark.unit
def test_update_different_scale():
    """Test Elo update with different scale parameter."""
    ra_old, rb_old = 1600.0, 1800.0  # B has advantage
    
    # Standard scale (400)
    ra_400, rb_400 = update(ra_old, rb_old, score_a=1.0, k=32.0, scale=400.0)
    
    # Smaller scale (200) - should make rating differences more impactful
    ra_200, rb_200 = update(ra_old, rb_old, score_a=1.0, k=32.0, scale=200.0)
    
    # With smaller scale, upset should create larger rating change
    change_400 = abs(ra_400 - ra_old)
    change_200 = abs(ra_200 - ra_old)
    
    assert change_200 > change_400


@pytest.mark.unit
def test_k_factor_thresholds():
    """Test K-factor thresholds based on rating."""
    # Low rating should get high K-factor
    assert k_factor(1000.0) == 40.0
    assert k_factor(2000.0) == 40.0
    
    # Medium rating should get medium K-factor
    assert k_factor(2100.0) == 20.0
    assert k_factor(2300.0) == 20.0
    
    # High rating should get low K-factor
    assert k_factor(2400.0) == 10.0
    assert k_factor(2800.0) == 10.0


@pytest.mark.unit
def test_k_factor_boundary_values():
    """Test K-factor at exact boundary values."""
    # Test boundary at 2100
    assert k_factor(2099.0) == 40.0
    assert k_factor(2100.0) == 20.0
    
    # Test boundary at 2400
    assert k_factor(2399.0) == 20.0
    assert k_factor(2400.0) == 10.0


@pytest.mark.unit
def test_k_factor_with_games():
    """Test K-factor with games parameter (currently unused)."""
    # Games parameter should not affect result in current implementation
    assert k_factor(1500.0, games=0) == 40.0
    assert k_factor(1500.0, games=100) == 40.0
    assert k_factor(2200.0, games=50) == 20.0
    assert k_factor(2500.0, games=200) == 10.0