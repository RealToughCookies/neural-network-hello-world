from __future__ import annotations
from typing import Dict, Any, Tuple, Iterable
from pathlib import Path
import re
import torch

# ---------- helpers ----------
def _subdict(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    p = prefix if prefix.endswith(".") else prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}

def _first_linear_in_dim(sd: Dict[str, torch.Tensor]) -> int | None:
    # heuristics: find a *.weight shaped [out, in] from the earliest-seeming layer
    candidates = [k for k in sd.keys() if k.endswith(".weight")]
    if not candidates: 
        return None
    # prefer net.0.weight or mlp.0.weight or 0.weight
    def score(k: str) -> Tuple[int,int]:
        m = re.search(r"(net|mlp|)(?:\.|^)(\d+)\.weight$", k)
        idx = int(m.group(2)) if m else 99
        pref = 0 if "net" in k else 1 if "mlp" in k else 2
        return (pref, idx)
    k = sorted(candidates, key=score)[0]
    w = sd[k]
    return int(w.shape[1]) if w.ndim == 2 else None

def _remap_inner_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Accept net.*, mlp.*, or bare '0.' style and normalize to net.*"""
    out = {}
    for k, v in sd.items():
        if k.startswith("net."):
            out[k] = v
        elif k.startswith("mlp."):
            out["net." + k[4:]] = v
        elif re.match(r"^\d+\.", k):
            out["net." + k] = v
        else:
            out[k] = v
    return out

def _normalize_ckpt_obj(ck: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"], ck.get("meta", {}) or {}
    if isinstance(ck, dict):
        for key in ("state_dict","policy","model_state_dict"):
            if key in ck and isinstance(ck[key], dict):
                return ck[key], ck.get("meta", {}) or {}
        if ck and all(isinstance(v, torch.Tensor) for v in ck.values()):
            return ck, {}
    raise ValueError("Unsupported checkpoint format")

# ---------- main API ----------
def save_checkpoint(model, meta: Dict[str, Any], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"model": model.state_dict(), "meta": meta}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); tmp.replace(path)

def load_policy_role_scoped(model, full_sd: Dict[str, torch.Tensor], expect_dims: Dict[str,int], roles=("good","adv")) -> None:
    """
    Load heads by role with strict checks, remapping inner prefixes if needed.
    Also supports legacy 'pi.0.*'/'pi.1.*' by inferring role from input dim.
    """
    # 1) discover role groups under pi./vf.
    def _group_children(prefix: str) -> Dict[str, Dict[str, torch.Tensor]]:
        # collect immediate children under prefix (e.g., pi.good.*, pi.0.*)
        kids = {}
        plen = len(prefix) + 1 if not prefix.endswith(".") else len(prefix)
        for k in full_sd.keys():
            if not k.startswith(prefix if prefix.endswith(".") else prefix + "."):
                continue
            rest = k[plen:]
            child = rest.split(".", 1)[0]
            kids.setdefault(child, {})
        for child in list(kids):
            kids[child] = _subdict(full_sd, f"{prefix}.{child}")
        return kids

    pi_children = _group_children("pi")
    vf_children = _group_children("vf")

    # 2) map child name -> role
    name_to_role: Dict[str,str] = {}
    if "good" in pi_children and "adv" in pi_children:
        name_to_role = {"good":"good","adv":"adv"}
    else:
        # legacy numeric keys; infer by in-dim
        for child, sd in pi_children.items():
            in_dim = _first_linear_in_dim(sd)
            if in_dim == expect_dims.get("good"):
                name_to_role[child] = "good"
            elif in_dim == expect_dims.get("adv"):
                name_to_role[child] = "adv"
        # sanity: both roles must be assigned
        if set(name_to_role.values()) != {"good","adv"}:
            raise RuntimeError(f"[loader] cannot infer role mapping from pi.* children {list(pi_children.keys())}")

    # 3) load each role strictly (after remapping inner prefixes)
    for child, role in name_to_role.items():
        pi_sd = _remap_inner_prefix(pi_children[child])
        vf_sd = _remap_inner_prefix(vf_children.get(child, {}))
        missing, unexpected = model.pi[role].load_state_dict(pi_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"[loader] pi.{role} mismatch: missing={missing}, unexpected={unexpected}")
        if vf_sd:
            missing, unexpected = model.vf[role].load_state_dict(vf_sd, strict=True)
            if missing or unexpected:
                raise RuntimeError(f"[loader] vf.{role} mismatch: missing={missing}, unexpected={unexpected}")

    # 4) load leftovers (shared) non-strict
    leftovers = {k: v for k, v in full_sd.items() if not (k.startswith("pi.") or k.startswith("vf."))}
    if leftovers:
        model.load_state_dict(leftovers, strict=False)

def load_policy_from_ckpt(model, ckpt: str | Path | dict, expect_dims: Dict[str,int]) -> Dict[str,int]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) if not isinstance(ckpt, dict) else ckpt
    sd, meta = _normalize_ckpt_obj(ck)
    saved_dims = meta.get("obs_dims") or expect_dims
    if dict(saved_dims) != dict(expect_dims):
        raise ValueError(f"Obs-dims mismatch: ckpt={saved_dims}, model={expect_dims}")
    load_policy_role_scoped(model, sd, expect_dims=expect_dims, roles=("good","adv"))
    return saved_dims

def load_legacy_checkpoint(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return _normalize_ckpt_obj(ck)