"""
Elo rating system utilities.
"""

import math


def expected_score(ra: float, rb: float, scale: float = 400.0) -> float:
    """
    Compute expected score for player A vs player B using Elo ratings.
    
    Args:
        ra: Rating of player A
        rb: Rating of player B  
        scale: Elo scale parameter (default 400.0)
        
    Returns:
        Expected score for player A (0.0 to 1.0)
    """
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))


def k_factor(rating: float, games: int = 0) -> float:
    """
    Compute K-factor based on rating and games played.
    Inspired by FIDE-style thresholds but simplified.
    
    Args:
        rating: Current Elo rating
        games: Number of games played (unused in this implementation)
        
    Returns:
        K-factor value
    """
    if rating >= 2400:
        return 10.0
    elif rating >= 2100:
        return 20.0  
    else:
        return 40.0


def update(ra: float, rb: float, score_a: float, k: float = 32.0, scale: float = 400.0) -> tuple[float, float]:
    """
    Update Elo ratings after a game.
    
    Args:
        ra: Current rating of player A
        rb: Current rating of player B
        score_a: Actual score for player A (1.0 = win, 0.5 = draw, 0.0 = loss)
        k: K-factor (rating change sensitivity, default 32.0)
        scale: Elo scale parameter (default 400.0)
        
    Returns:
        Tuple of (new_ra, new_rb)
    """
    expected_a = expected_score(ra, rb, scale)
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    
    new_ra = ra + k * (score_a - expected_a)
    new_rb = rb + k * (score_b - expected_b)
    
    return new_ra, new_rb