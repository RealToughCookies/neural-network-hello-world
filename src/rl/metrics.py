"""
Statistical metrics and confidence intervals for RL evaluation.
"""

import math


def wilson_ci(wins: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Compute Wilson score confidence interval for binomial proportion.
    
    This is more reliable than normal approximation for small sample sizes
    and proportions near 0 or 1.
    
    Args:
        wins: Number of successes/wins
        n: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)
        
    Returns:
        Tuple of (lower_bound, upper_bound) confidence interval
        
    References:
        Wilson, E.B. (1927). "Probable inference, the law of succession, 
        and statistical inference". Journal of the American Statistical Association.
    """
    if n == 0:
        return 0.0, 1.0
    
    # Normal distribution critical value for alpha/2
    z_alpha_2 = {
        0.01: 2.576,   # 99% CI
        0.05: 1.96,    # 95% CI  
        0.10: 1.645,   # 90% CI
    }.get(alpha, 1.96)  # Default to 95% CI
    
    p = wins / n
    z_sq = z_alpha_2 * z_alpha_2
    
    # Wilson score interval formula
    denominator = 1 + z_sq / n
    center = (p + z_sq / (2 * n)) / denominator
    margin = z_alpha_2 * math.sqrt((p * (1 - p) + z_sq / (4 * n)) / n) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return lower, upper