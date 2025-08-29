import subprocess, sys, re
import pytest

@pytest.mark.smoke
@pytest.mark.rl
def test_dota_budget_128(tmp_path):
    cmd = [sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
           "--train", "--selfplay", "--env", "dota_last_hit",
           "--steps", "50", "--eval-every", "1",
           "--save-dir", str(tmp_path / "ck"),
           "--rollout-steps", "128", "--seed", "123", "--updates", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    # Check that training completed successfully with proper outputs
    assert "Smoke train ✓ PASSED" in out.stdout, out.stdout
    assert "[ppo]" in out.stdout, "PPO training should have occurred"
    assert "[ckpt v3]" in out.stdout, "Should save v3 checkpoints"