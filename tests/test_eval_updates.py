import pytest
import subprocess
import sys
import json
import tempfile
from pathlib import Path


@pytest.mark.cli
def test_eval_pool_update_flags():
    """Test eval_cli with pool update flags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir()
        
        # Create minimal checkpoint by running quick training
        cmd_train = [
            sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
            "--train", "--selfplay", "--env", "dota_last_hit",
            "--steps", "10", "--rollout-steps", "32", "--seed", "42",
            "--save-dir", str(ckpt_dir), "--updates", "1"
        ]
        train_out = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
        assert train_out.returncode == 0, f"Failed to create test checkpoint: {train_out.stderr}"
        
        # Create empty pool
        pool_path = Path(tmpdir) / "test_pool.json"
        
        # Test basic pool update without write-agent
        cmd = [
            sys.executable, "-m", "src.rl.eval_cli",
            "--ckpt", str(ckpt_dir),
            "--episodes", "2",
            "--env", "dota_last_hit", 
            "--pool-path-v2", str(pool_path),
            "--pool-update",
            "--agent-policy-id", "test_agent"
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Should fail because agent doesn't exist in pool yet
        assert out.returncode != 0 or "Agent not found" in out.stderr or "no agents" in out.stdout


@pytest.mark.cli
def test_eval_write_agent():
    """Test eval_cli with --write-agent flag.""" 
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir()
        
        cmd_train = [
            sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
            "--train", "--selfplay", "--env", "dota_last_hit",
            "--steps", "10", "--rollout-steps", "32", "--seed", "42",
            "--save-dir", str(ckpt_dir), "--updates", "1"
        ]
        train_out = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
        assert train_out.returncode == 0, f"Failed to create test checkpoint: {train_out.stderr}"
        
        pool_path = Path(tmpdir) / "test_pool.json"
        
        # Test with write-agent (should auto-register)
        cmd = [
            sys.executable, "-m", "src.rl.eval_cli",
            "--ckpt", str(ckpt_dir),
            "--episodes", "2",
            "--env", "dota_last_hit",
            "--pool-path-v2", str(pool_path),
            "--pool-update",
            "--write-agent",
            "--agent-elo", "1100"
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Should succeed and auto-register
        assert out.returncode == 0, f"eval_cli failed: {out.stderr}\nStdout: {out.stdout}"
        assert "auto-registered agent" in out.stdout
        
        # Check pool was created and agent registered  
        assert pool_path.exists()
        with open(pool_path) as f:
            pool = json.load(f)
        
        assert pool["schema"] == 2
        assert len(pool["agents"]) == 1
        assert pool["agents"][0]["elo"] == 1100.0
        assert pool["agents"][0]["role"] == "good"


@pytest.mark.cli 
def test_eval_pool_updates_with_matches():
    """Test that matches are recorded when pool updates are enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir()
        
        cmd_train = [
            sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
            "--train", "--selfplay", "--env", "dota_last_hit",
            "--steps", "10", "--rollout-steps", "32", "--seed", "42",
            "--save-dir", str(ckpt_dir), "--updates", "1"
        ]
        train_out = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
        assert train_out.returncode == 0
        
        # Create pool with two agents manually
        pool_path = Path(tmpdir) / "test_pool.json"
        pool = {
            "schema": 2,
            "env": "dota_last_hit",
            "roles": ["good"],
            "agents": [
                {
                    "policy_id": "agent_a",
                    "role": "good",
                    "ckpt": str(ckpt_dir / "last.pt"),
                    "elo": 1000.0,
                    "games": 0,
                    "wins": 0,
                    "created": "2024-01-01T00:00:00Z"
                },
                {
                    "policy_id": "agent_b",
                    "role": "good", 
                    "ckpt": str(ckpt_dir / "last.pt"),
                    "elo": 1200.0,
                    "games": 0,
                    "wins": 0,
                    "created": "2024-01-01T00:00:00Z"
                }
            ],
            "matches": []
        }
        
        with open(pool_path, 'w') as f:
            json.dump(pool, f)
        
        # Run eval with pool updates
        cmd = [
            sys.executable, "-m", "src.rl.eval_cli",
            "--ckpt", str(ckpt_dir),
            "--episodes", "4",  # Should sample both agents
            "--env", "dota_last_hit",
            "--pool-path-v2", str(pool_path),
            "--pool-update",
            "--agent-policy-id", "agent_a",
            "--pool-strategy", "uniform"
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        assert out.returncode == 0, f"eval_cli failed: {out.stderr}\nStdout: {out.stdout}"
        assert "Pool updated with" in out.stdout
        assert "saved updates to" in out.stdout
        
        # Check that matches were recorded
        with open(pool_path) as f:
            updated_pool = json.load(f)
        
        # Should have match records
        assert len(updated_pool["matches"]) > 0
        
        # Agent games should be incremented
        agent_a = next(a for a in updated_pool["agents"] if a["policy_id"] == "agent_a")
        assert agent_a["games"] > 0


@pytest.mark.cli
def test_eval_agent_policy_id_generation():
    """Test automatic policy ID generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint with specific name
        ckpt_dir = Path(tmpdir) / "my_model_v2"
        ckpt_dir.mkdir()
        
        cmd_train = [
            sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
            "--train", "--selfplay", "--env", "dota_last_hit",
            "--steps", "10", "--rollout-steps", "32", "--seed", "42",
            "--save-dir", str(ckpt_dir), "--updates", "1"
        ]
        train_out = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
        assert train_out.returncode == 0
        
        pool_path = Path(tmpdir) / "test_pool.json"
        
        # Test auto-generation of policy ID from directory name
        cmd = [
            sys.executable, "-m", "src.rl.eval_cli",
            "--ckpt", str(ckpt_dir),
            "--episodes", "1",
            "--env", "dota_last_hit",
            "--pool-path-v2", str(pool_path),
            "--pool-update",
            "--write-agent"
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        assert out.returncode == 0, f"eval_cli failed: {out.stderr}\nStdout: {out.stdout}"
        
        # Check generated policy ID
        with open(pool_path) as f:
            pool = json.load(f)
        
        # Should generate policy ID from directory name
        agent_ids = [a["policy_id"] for a in pool["agents"]]
        assert "my_model_v2:learner" in agent_ids


@pytest.mark.cli
def test_eval_no_pool_update_no_changes():
    """Test that without --pool-update, no changes are made to pool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint
        ckpt_dir = Path(tmpdir) / "ckpt"
        ckpt_dir.mkdir()
        
        cmd_train = [
            sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
            "--train", "--selfplay", "--env", "dota_last_hit",
            "--steps", "10", "--rollout-steps", "32", "--seed", "42",
            "--save-dir", str(ckpt_dir), "--updates", "1"
        ]
        train_out = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
        assert train_out.returncode == 0
        
        # Create pool with agent
        pool_path = Path(tmpdir) / "test_pool.json"
        initial_pool = {
            "schema": 2,
            "env": "dota_last_hit",
            "roles": ["good"],
            "agents": [
                {
                    "policy_id": "test_agent",
                    "role": "good",
                    "ckpt": str(ckpt_dir / "last.pt"),
                    "elo": 1000.0,
                    "games": 0,
                    "wins": 0,
                    "created": "2024-01-01T00:00:00Z"
                }
            ],
            "matches": []
        }
        
        with open(pool_path, 'w') as f:
            json.dump(initial_pool, f)
        
        # Run eval WITHOUT pool-update
        cmd = [
            sys.executable, "-m", "src.rl.eval_cli",
            "--ckpt", str(ckpt_dir),
            "--episodes", "2",
            "--env", "dota_last_hit",
            "--pool-path-v2", str(pool_path),
            "--pool-strategy", "uniform"
            # Note: NO --pool-update flag
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        assert out.returncode == 0
        
        # Pool should be unchanged
        with open(pool_path) as f:
            final_pool = json.load(f)
        
        # Should be identical to initial state
        assert final_pool["agents"][0]["games"] == 0
        assert final_pool["agents"][0]["elo"] == 1000.0
        assert len(final_pool["matches"]) == 0
        assert "Pool updated" not in out.stdout
        assert "saved updates" not in out.stdout


@pytest.mark.cli
def test_eval_custom_agent_elo():
    """Test setting custom agent Elo rating."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test checkpoint
        ckpt_dir = Path(tmpdir) / "ckpt" 
        ckpt_dir.mkdir()
        
        cmd_train = [
            sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
            "--train", "--selfplay", "--env", "dota_last_hit",
            "--steps", "10", "--rollout-steps", "32", "--seed", "42",
            "--save-dir", str(ckpt_dir), "--updates", "1"
        ]
        train_out = subprocess.run(cmd_train, capture_output=True, text=True, timeout=60)
        assert train_out.returncode == 0
        
        pool_path = Path(tmpdir) / "test_pool.json"
        
        # Test with custom agent Elo
        cmd = [
            sys.executable, "-m", "src.rl.eval_cli",
            "--ckpt", str(ckpt_dir),
            "--episodes", "1",
            "--env", "dota_last_hit",
            "--pool-path-v2", str(pool_path),
            "--pool-update",
            "--write-agent",
            "--agent-elo", "1500"
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        assert out.returncode == 0
        
        # Check agent has custom Elo
        with open(pool_path) as f:
            pool = json.load(f)
        
        assert len(pool["agents"]) == 1
        assert pool["agents"][0]["elo"] == 1500.0