import random
import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = False):
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enable PyTorch deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # Some backends may not support strict determinism