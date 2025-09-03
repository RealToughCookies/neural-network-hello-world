"""
Reproducibility utilities for RL experiments.

Sets global seeds across Python, NumPy, PyTorch for deterministic runs.
Based on PyTorch documentation for reproducible results.
"""

def set_global_seed(seed: int, deterministic: bool = False):
    """
    Set global random seed across all libraries for reproducible experiments.
    
    Args:
        seed: Random seed to use across all libraries
        deterministic: Enable PyTorch deterministic algorithms (slower but fully reproducible)
    
    Note:
        When deterministic=True, some operations may be slower or unsupported.
        See PyTorch docs: https://pytorch.org/docs/stable/notes/randomness.html
    """
    import os
    import random
    import numpy as np
    import torch
    
    # Python hash seed for dictionary ordering
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Standard library and NumPy
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch CPU and GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms for full reproducibility
    if deterministic:
        torch.use_deterministic_algorithms(True)