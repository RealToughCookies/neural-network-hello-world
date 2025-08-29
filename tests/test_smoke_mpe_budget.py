import subprocess, sys, re, os
import pytest

@pytest.mark.smoke
@pytest.mark.rl
def test_mpe_budget_128(tmp_path, monkeypatch):
    # headless pygame
    monkeypatch.setenv("SDL_VIDEODRIVER", "null")
    cmd = [sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
           "--train", "--selfplay", "--env", "mpe_adversary",
           "--steps", "30", "--rollout-steps", "128", "--seed", "42", "--updates", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    assert out.returncode == 0, out.stderr
    # Check that training completed successfully (MPE environment works)
    assert "[ppo]" in out.stdout, "PPO training should have occurred"
    assert "[ckpt v3]" in out.stdout, "Should save v3 checkpoints"
    assert "good" in out.stdout and "adv" in out.stdout, "Should show role info"