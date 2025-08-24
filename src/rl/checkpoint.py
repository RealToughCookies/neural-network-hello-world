#!/usr/bin/env python3
"""
Checkpoint utilities for role-aware model loading and saving.
Ensures safe loading of multi-role models with dimension validation.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


def subdict_with_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Extract subdict with given prefix, removing prefix from keys."""
    p = prefix if prefix.endswith('.') else prefix + '.'
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


def load_policy_from_ckpt(model: nn.Module, ckpt: Dict[str, Any], 
                         expect_dims: Dict[str, int], strict: bool = True) -> Dict[str, int]:
    """
    Load policy from checkpoint with role-aware dimension validation.
    
    Args:
        model: MultiHeadPolicy model with pi/vf ModuleDicts
        ckpt: Checkpoint dictionary with 'model' and 'meta' keys
        expect_dims: Expected observation dimensions per role
        strict: Whether to strictly validate all keys match
        
    Returns:
        saved_dims: Observation dimensions from checkpoint metadata
        
    Raises:
        ValueError: If dimensions don't match
        RuntimeError: If state dict loading fails
    """
    # Validate checkpoint has required metadata
    meta = ckpt.get("meta", {})
    saved_dims = meta.get("obs_dims")
    if not saved_dims:
        raise ValueError("Checkpoint missing meta.obs_dims; re-train checkpoint after role-dim patch.")
    
    # Validate dimensions match
    if saved_dims != expect_dims:
        raise ValueError(f"Obs-dims mismatch: ckpt={saved_dims}, model={expect_dims}")
    
    full_state_dict = ckpt["model"]
    
    # Load role heads by prefix for safety
    for role in ["good", "adv"]:
        if role not in expect_dims:
            continue
            
        # Extract policy head state dict
        pi_sd = subdict_with_prefix(full_state_dict, f"pi.{role}")
        if pi_sd:
            missing, unexpected = model.pi[role].load_state_dict(pi_sd, strict=strict)
            if missing or unexpected:
                raise RuntimeError(f"Policy head '{role}' keys mismatch: missing={missing}, unexpected={unexpected}")
        
        # Extract value head state dict
        vf_sd = subdict_with_prefix(full_state_dict, f"vf.{role}")
        if vf_sd:
            missing, unexpected = model.vf[role].load_state_dict(vf_sd, strict=strict)
            if missing or unexpected:
                raise RuntimeError(f"Value head '{role}' keys mismatch: missing={missing}, unexpected={unexpected}")
    
    # Load any shared parts (if you have them) AFTER heads, by filtering out pi./vf. keys
    shared = {k: v for k, v in full_state_dict.items() if not (k.startswith("pi.") or k.startswith("vf."))}
    if shared:
        model.load_state_dict(shared, strict=False)
    
    return saved_dims


def load_legacy_checkpoint(pi_good: nn.Module, pi_adv: nn.Module, 
                          vf_good: nn.Module, vf_adv: nn.Module,
                          ckpt: Dict[str, Any]) -> None:
    """
    Load legacy checkpoint format with separate pi_good, pi_adv, vf_good, vf_adv keys.
    For backward compatibility with existing checkpoints.
    
    Args:
        pi_good: Policy head for good role
        pi_adv: Policy head for adversary role  
        vf_good: Value head for good role
        vf_adv: Value head for adversary role
        ckpt: Legacy checkpoint dictionary
    """
    if "pi_good" in ckpt:
        pi_good.load_state_dict(ckpt["pi_good"])
    if "pi_adv" in ckpt:
        pi_adv.load_state_dict(ckpt["pi_adv"])
    if "vf_good" in ckpt:
        vf_good.load_state_dict(ckpt["vf_good"])
    if "vf_adv" in ckpt:
        vf_adv.load_state_dict(ckpt["vf_adv"])


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   step: int, obs_dims: Dict[str, int], config: Dict[str, Any],
                   meta: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Save checkpoint with proper metadata for role-aware loading.
    
    Args:
        model: MultiHeadPolicy model
        optimizer: Optimizer state
        step: Training step
        obs_dims: Observation dimensions per role
        config: Training configuration
        meta: Additional metadata
        
    Returns:
        checkpoint: Dictionary ready for torch.save
    """
    if meta is None:
        meta = {}
    
    meta.update({
        "obs_dims": obs_dims,
        "n_act": getattr(model, 'n_actions', 5),
        "role_map": {"adversary": "adv", "agent": "good"}  # Standard role mapping
    })
    
    checkpoint = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "config": config,
        "meta": meta
    }
    
    return checkpoint


def load_opponent_role(model: nn.Module, ckpt: Dict[str, Any], 
                      opponent_role: str, expect_dims: Dict[str, int]) -> None:
    """
    Load only the opponent role from checkpoint, leaving learner role unchanged.
    Used for loading opponent policies during training.
    
    Args:
        model: MultiHeadPolicy model
        ckpt: Checkpoint dictionary
        opponent_role: Role to load ("good" or "adv")
        expect_dims: Expected dimensions for validation
    """
    # Validate checkpoint has required metadata
    meta = ckpt.get("meta", {})
    saved_dims = meta.get("obs_dims")
    if saved_dims and saved_dims != expect_dims:
        raise ValueError(f"Opponent checkpoint dims mismatch: ckpt={saved_dims}, expected={expect_dims}")
    
    full_state_dict = ckpt["model"]
    
    # Load only the opponent role
    pi_sd = subdict_with_prefix(full_state_dict, f"pi.{opponent_role}")
    if pi_sd:
        model.pi[opponent_role].load_state_dict(pi_sd)
    
    vf_sd = subdict_with_prefix(full_state_dict, f"vf.{opponent_role}")
    if vf_sd:
        model.vf[opponent_role].load_state_dict(vf_sd)