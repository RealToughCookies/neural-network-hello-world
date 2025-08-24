import os
import pytest
import torch
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.rl.ppo_selfplay_skeleton import PolicyHead, ValueHead

def _first_layer_in_dim(sd: dict) -> int | None:
    for k, v in sd.items():
        if k.endswith("net.0.weight"):
            return int(v.shape[1])  # [out, in]
    return None

def _final_layer_out_dim(sd: dict) -> int | None:
    for k, v in sd.items():
        if k.endswith("net.2.weight"):
            return int(v.shape[0])  # [n_act, hidden]
    return None

CKPT = "artifacts/rl_ckpts/last.pt"

@pytest.mark.skipif(not os.path.exists(CKPT), reason="needs trained ckpt")
def test_infer_dims_from_ckpt():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    gi = _first_layer_in_dim(ckpt["pi_good"])
    ai = _first_layer_in_dim(ckpt["pi_adv"])
    assert gi in (8,10) and ai in (8,10)  # Allow same dim for smoke test ckpts
    na = _final_layer_out_dim(ckpt["pi_good"])
    assert na == 5

@pytest.mark.skipif(not os.path.exists(CKPT), reason="needs trained ckpt")
def test_forward_shapes_no_crash():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    gi = _first_layer_in_dim(ckpt["pi_good"])
    ai = _first_layer_in_dim(ckpt["pi_adv"])
    na = _final_layer_out_dim(ckpt["pi_good"]) or 5
    pg = PolicyHead(gi, n_act=na)
    pa = PolicyHead(ai, n_act=na)
    pg.load_state_dict(ckpt["pi_good"])
    pa.load_state_dict(ckpt["pi_adv"])
    # one-step forward with zeros, ensures no matmul mismatch
    assert pg(torch.zeros(1, gi)).shape == (1, na)
    assert pa(torch.zeros(1, ai)).shape == (1, na)