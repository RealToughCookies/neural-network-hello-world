"""
Experiment manifest generation for reproducibility tracking.

Creates JSON manifests with git state, dependencies, and run configuration.
"""

import json
import subprocess
import sys
import datetime
import torch
import pettingzoo


def write_manifest(path: str, args, device: torch.device):
    """
    Write experiment manifest with reproducibility information.
    
    Args:
        path: Output path for manifest.json
        args: Parsed command line arguments
        device: PyTorch device being used
    
    Captures:
        - Git commit SHA for code version tracking
        - Full command line arguments
        - Library versions (PyTorch, PettingZoo)
        - Device and seed configuration
        - UTC timestamp
    """
    try:
        # Get current git commit
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sha = "unknown"
    
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "git_sha": sha,
        "argv": sys.argv,
        "env": args.env,
        "steps": getattr(args, "steps", None),
        "pz_version": getattr(pettingzoo, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": getattr(args, "global_seed", None),
        "deterministic": getattr(args, "deterministic", False),
    }
    
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)