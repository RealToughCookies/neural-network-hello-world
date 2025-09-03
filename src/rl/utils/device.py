"""
Device selection utilities for PyTorch.

Automatically picks the best available device: MPS (Apple Silicon) > CUDA > CPU.
Based on PyTorch MPS backend documentation.
"""

import torch


def pick_device():
    """
    Pick the best available PyTorch device.
    
    Preference order:
    1. MPS (Apple Silicon GPU) - Official Mac GPU backend
    2. CUDA (NVIDIA GPU) 
    3. CPU (fallback)
    
    Returns:
        torch.device: Best available device
    
    References:
        - PyTorch MPS: https://pytorch.org/docs/stable/notes/mps.html
        - Apple Developer: https://developer.apple.com/metal/pytorch/
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    
    return torch.device("cpu")  # CPU fallback