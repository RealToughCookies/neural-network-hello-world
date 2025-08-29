import json, os, tempfile, torch
import pytest
from pathlib import Path
from src.rl.ckpt_io import save_checkpoint_v3, load_checkpoint_auto

@pytest.mark.unit
def test_save_and_load_v3_bundle_and_weights(tmp_path: Path):
    fake_sd = {"foo.weight": torch.zeros(1)}
    bundle = {"version": 3, "model_state": fake_sd, "counters": {"global_step": 123}, "meta": {"seed": 1337}}
    save_dir = tmp_path
    save_checkpoint_v3(bundle, save_dir)

    # both files exist atomically
    assert (save_dir / "last.pt").exists()
    assert (save_dir / "last_model.pt").exists()

    kind, payload = load_checkpoint_auto(save_dir)  # dir fallback selects last.pt
    assert kind == "v3"
    assert payload["counters"]["global_step"] == 123
    assert list(payload["model_state"].keys()) == ["foo.weight"]

    # model_only fallback works too
    kind2, weights = load_checkpoint_auto(save_dir / "last_model.pt")
    assert kind2 == "model_only"
    assert "foo.weight" in weights