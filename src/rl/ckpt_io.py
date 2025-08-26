"""
v3 Checkpoint I/O System

Provides comprehensive checkpoint management with atomic saves, RNG state preservation,
and automatic resume logic for PPO training and evaluation.
"""

import os
import time
import random
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np


def _git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _iso8601_now() -> str:
    """Get current timestamp in ISO8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state from all random sources."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate()
    }
    
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
        
    return state


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    """Restore RNG state to all random sources."""
    torch.set_rng_state(rng_state["torch"])
    np.random.set_state(rng_state["numpy"])
    random.setstate(rng_state["python"])
    
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def make_bundle(
    *,
    version: int = 3,
    model_state: Dict[str, torch.Tensor],
    optim_state: Dict[str, Any],
    sched_state: Optional[Dict[str, Any]],
    obs_norm_state: Dict[str, Dict[str, Any]],
    adv_norm_state: Optional[Dict[str, Any]],
    rng_state: Dict[str, Any],
    counters: Dict[str, int],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a v3 checkpoint bundle with complete training state.
    
    Args:
        version: Bundle format version (should be 3)
        model_state: model.state_dict()
        optim_state: optimizer.state_dict()
        sched_state: scheduler.state_dict() or None
        obs_norm_state: {role: normalizer.state_dict() for role in normalizers}
        adv_norm_state: advantage_normalizer.state_dict() or None
        rng_state: RNG states from _capture_rng_state()
        counters: {"global_step": int, "update_idx": int, "episodes": int}
        meta: {"created_at": str, "env": str, "adapter": str, "git_commit": str, "seed": int}
    
    Returns:
        Complete v3 bundle dictionary
    """
    if version != 3:
        raise ValueError(f"Only version=3 bundles supported, got {version}")
    
    bundle = {
        "version": version,
        "created_at": _iso8601_now(),
        "model_state": model_state,
        "optim_state": optim_state,
        "sched_state": sched_state,
        "obs_norm_state": obs_norm_state,
        "adv_norm_state": adv_norm_state,
        "rng_state": rng_state,
        "counters": counters,
        "meta": meta
    }
    
    return bundle


def save_checkpoint_v3(bundle: Dict[str, Any], save_dir: Path) -> Tuple[Path, Path]:
    """
    Atomically save v3 bundle and extract model-only checkpoint.
    
    Args:
        bundle: v3 bundle from make_bundle()
        save_dir: Directory to save checkpoints
        
    Returns:
        Tuple of (bundle_path, model_path) where files were saved
    """
    if bundle.get("version") != 3:
        raise ValueError(f"Expected version=3 bundle, got {bundle.get('version')}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    bundle_path = save_dir / "last.pt"
    model_path = save_dir / "last_model.pt"
    
    # Atomic save of v3 bundle
    bundle_tmp = bundle_path.with_suffix(".tmp")
    torch.save(bundle, bundle_tmp)
    os.replace(bundle_tmp, bundle_path)
    
    # Extract and save model-only checkpoint
    model_only = {
        "model": bundle["model_state"],
        "meta": bundle["meta"]
    }
    model_tmp = model_path.with_suffix(".tmp")
    torch.save(model_only, model_tmp)
    os.replace(model_tmp, model_path)
    
    return bundle_path, model_path


def load_checkpoint_auto(path_or_dir: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Load checkpoint automatically from file or directory.
    
    Args:
        path_or_dir: Path to checkpoint file or directory containing checkpoints
        
    Returns:
        Tuple of (kind, obj) where:
        - kind: "v3" for v3 bundles, "model_only" for model-only checkpoints
        - obj: Loaded checkpoint data
        
    Raises:
        FileNotFoundError: If no suitable checkpoint found
        ValueError: If checkpoint format is invalid
    """
    path_or_dir = Path(path_or_dir)
    
    if path_or_dir.is_dir():
        # Directory provided - auto-select checkpoint
        candidates = [
            path_or_dir / "last.pt",
            path_or_dir / "last_model.pt"
        ]
        
        ckpt_path = None
        for candidate in candidates:
            if candidate.exists():
                ckpt_path = candidate
                break
                
        if ckpt_path is None:
            available = list(path_or_dir.glob("*.pt"))
            raise FileNotFoundError(
                f"No suitable checkpoint found in {path_or_dir}. "
                f"Available: {[p.name for p in available]}"
            )
    else:
        # File provided directly
        if not path_or_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path_or_dir}")
        ckpt_path = path_or_dir
    
    # Load checkpoint
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Determine checkpoint kind
    if isinstance(obj, dict) and obj.get("version") == 3:
        # v3 bundle
        required_keys = [
            "model_state", "optim_state", "rng_state", "counters", "meta"
        ]
        missing = [k for k in required_keys if k not in obj]
        if missing:
            raise ValueError(f"Invalid v3 bundle, missing keys: {missing}")
        return "v3", obj
    
    elif isinstance(obj, dict) and "model" in obj:
        # Model-only checkpoint (v2 style or extracted from v3)
        return "model_only", obj
    
    else:
        # Raw state dict or unknown format
        return "model_only", {"model": obj, "meta": {}}


def restore_rng_state(bundle: Dict[str, Any]) -> None:
    """Restore RNG state from v3 bundle."""
    if bundle.get("version") != 3:
        raise ValueError(f"Expected version=3 bundle, got {bundle.get('version')}")
    
    _restore_rng_state(bundle["rng_state"])


def capture_rng_state() -> Dict[str, Any]:
    """Capture current RNG state for saving in bundle."""
    return _capture_rng_state()