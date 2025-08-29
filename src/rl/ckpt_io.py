"""
v3 Checkpoint I/O System

v3 bundle schema:
{
    "version": 3,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "sched_state": scheduler.state_dict() or None,
    "obs_norm_state": {role: normalizer.state_dict() for role in normalizers} or None,
    "adv_norm_state": adv_normalizer.state_dict() or None,
    "rng_state": {"torch_cpu": ..., "torch_cuda": ..., "numpy": ..., "python": ...},
    "counters": {"global_step": int, "update_idx": int, "episodes": int},
    "meta": {"created_at": str, "env": str, "adapter": str, "seed": int, "git_commit": str}
}
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import torch
import numpy as np


def _capture_rng() -> Dict[str, Any]:
    """Capture RNG state from all sources."""
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(s: Dict[str, Any]) -> None:
    """Restore RNG state to all sources."""
    torch.set_rng_state(s["torch_cpu"])
    if torch.cuda.is_available() and s.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(s["torch_cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def make_bundle(
    *,
    version: int,
    model_state: Dict[str, torch.Tensor],
    optim_state: Dict[str, Any],
    sched_state: Union[Dict[str, Any], None],
    obs_norm_state: Union[Dict[str, Dict[str, Any]], None],
    adv_norm_state: Union[Dict[str, Any], None],
    rng_state: Dict[str, Any],
    counters: Dict[str, int],
    meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a v3 checkpoint bundle with complete training state."""
    return {
        "version": version,
        "model_state": model_state,
        "optim_state": optim_state,
        "sched_state": sched_state,
        "obs_norm_state": obs_norm_state,
        "adv_norm_state": adv_norm_state,
        "rng_state": rng_state,
        "counters": counters,
        "meta": meta
    }


def save_checkpoint_v3(bundle: dict, save_dir: Path) -> tuple[Path, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    # 1) Full bundle → last.pt (atomic)
    bundle_path_tmp = save_dir / "last.pt.tmp"
    bundle_path     = save_dir / "last.pt"
    torch.save(bundle, bundle_path_tmp)
    os.replace(bundle_path_tmp, bundle_path)

    # 2) Weights-only → last_model.pt (atomic)
    model_only = bundle["model_state"]
    model_path_tmp = save_dir / "last_model.pt.tmp"
    model_path     = save_dir / "last_model.pt"
    torch.save(model_only, model_path_tmp)
    os.replace(model_path_tmp, model_path)

    return bundle_path, model_path


def load_checkpoint_auto(path_or_dir: Path, map_location: str = "cpu") -> Tuple[str, Union[Dict[str, Any], Dict[str, torch.Tensor]]]:
    """
    If directory: try last.pt → last_model.pt → best.pt.
    If file: inspect object:
      - dict with bundle["version"]==3 → ("v3", bundle)
      - Tensor/dict of weights → ("model_only", state_dict)
    """
    path_or_dir = Path(path_or_dir)
    
    if path_or_dir.is_dir():
        # Directory provided - try in preference order
        candidates = ["last.pt", "last_model.pt", "best.pt"]
        ckpt_path = None
        for candidate in candidates:
            candidate_path = path_or_dir / candidate
            if candidate_path.exists():
                ckpt_path = candidate_path
                break
                
        if ckpt_path is None:
            raise FileNotFoundError(f"No suitable checkpoint found in {path_or_dir}")
    else:
        # File provided directly
        if not path_or_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path_or_dir}")
        ckpt_path = path_or_dir
    
    # Load checkpoint
    obj = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    
    # Determine checkpoint kind
    if isinstance(obj, dict) and obj.get("version") == 3:
        # v3 bundle
        return "v3", obj
    else:
        # Model-only checkpoint or raw state dict
        return "model_only", obj