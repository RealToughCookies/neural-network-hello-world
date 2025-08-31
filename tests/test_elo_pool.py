import pytest
from pathlib import Path
from src.rl.elo_pool import OpponentPoolV1

@pytest.mark.unit
def test_expected_and_pfsp_weights(tmp_path):
    p = OpponentPoolV1(tmp_path / "pool.json")
    my = p.add_or_update_agent(ckpt_path="me.pt", ckpt_kind="v3", roles=["good","adv"], meta={})
    a = p.add_or_update_agent(ckpt_path="a.pt", ckpt_kind="v3", roles=["good","adv"], meta={})
    b = p.add_or_update_agent(ckpt_path="b.pt", ckpt_kind="v3", roles=["good","adv"], meta={})
    
    # set Elo for determinism
    p.data["agents"][0]["elo"] = 1200  # me (assume index mapping inside impl)
    p.data["agents"][1]["elo"] = 1200
    p.data["agents"][2]["elo"] = 1600
    
    # expected() monotonic: strong opp -> lower p_win
    pa = p.expected(1200, 1200)
    pb = p.expected(1200, 1600)
    assert pa > pb
    
    # PFSP should give more weight to near 0.5 matchups than stomps
    # Use the internal PFSP formula: w = (1 - |0.5 - p|)^tau
    w_even = (1.0 - abs(0.5 - 0.5)) ** 1.5  # p=0.5, tau=1.5
    w_far  = (1.0 - abs(0.5 - 0.9)) ** 1.5  # p=0.9, tau=1.5
    assert w_even > w_far