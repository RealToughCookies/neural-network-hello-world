import json, subprocess, sys
import pytest
from pathlib import Path

@pytest.mark.cli
def test_eval_dir_v3_empty_pool(tmp_path: Path):
    # Create a real checkpoint by running a quick training
    ckpt_dir = tmp_path / "ck"
    cmd_train = [sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
                 "--train", "--selfplay", "--env", "dota_last_hit",
                 "--steps", "10", "--rollout-steps", "32", "--seed", "42",
                 "--save-dir", str(ckpt_dir), "--updates", "1"]
    train_out = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
    assert train_out.returncode == 0, "Failed to create test checkpoint"

    # Create empty pool
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"version":"v1-elo-pool","config":{"initial":1200,"elo_k":32,"scale":400},"agents":[]}))

    # Test eval_cli with empty pool
    cmd = [sys.executable, "-m", "src.rl.eval_cli", "--ckpt", str(ckpt_dir),
           "--episodes", "2", "--env", "dota_last_hit", "--pool-path", str(pool)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, f"eval_cli failed: {out.stderr}\nStdout: {out.stdout}"
    assert "Directory provided, auto-selected: last.pt" in out.stdout
    assert "Skipping pool buckets (no agents)." in out.stdout