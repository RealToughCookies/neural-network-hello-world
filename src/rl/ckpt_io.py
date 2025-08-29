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


def save_checkpoint_v3(bundle: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    # last.pt (bundle)
    tmp = save_dir / "last.pt.tmp"
    out = save_dir / "last.pt"
    torch.save(bundle, tmp)
    os.replace(tmp, out)

    # last_model.pt (weights-only)
    model_only = bundle["model_state"]
    tmp = save_dir / "last_model.pt.tmp"
    out = save_dir / "last_model.pt"
    torch.save(model_only, tmp)
    os.replace(tmp, out)

    return


def load_checkpoint_auto(path_or_dir: Path, map_location="cpu"):
    p = Path(path_or_dir)
    if p.is_dir():
        for name in ("last.pt", "last_model.pt", "best.pt"):
            cand = p / name
            if cand.exists():
                print(f"Directory provided, auto-selected: {cand.name}")
                p = cand
                break
        else:
            raise FileNotFoundError(f"No checkpoint files found in {p}")

    obj = torch.load(p, map_location=map_location, weights_only=False)

    # v3 bundle detection (support multiple key names for BC)
    if isinstance(obj, dict) and (
        obj.get("version") == 3 or "model" in obj or "model_state" in obj
    ):
        # normalize shape
        weights = obj.get("model_state") or obj.get("model") or obj.get("state_dict")
        bundle = {
            "version": obj.get("version", 3),
            "model_state": weights,
            "optim_state": obj.get("optim_state"),
            "sched_state": obj.get("sched_state"),
            "obs_norm_state": obj.get("obs_norm_state"),
            "adv_norm_state": obj.get("adv_norm_state"),
            "rng_state": obj.get("rng_state"),
            "counters": obj.get("counters", {}),
            "meta": obj.get("meta", {}),
        }
        return "v3", bundle

    # plain weights (model-only)
    if isinstance(obj, dict):
        # common cases: a raw state_dict
        return "model_only", obj
    raise ValueError(f"Unsupported checkpoint format at {p}")