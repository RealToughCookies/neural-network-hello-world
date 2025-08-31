import json, subprocess, sys
import pytest
from pathlib import Path

@pytest.mark.cli
def test_migrate_pool_creates_v1(tmp_path: Path):
    pool = tmp_path / "pool.json"
    cmd = [sys.executable, "-m", "scripts.migrate_pool_schema", "--pool-path", str(pool), "--inplace"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0
    obj = json.loads(pool.read_text())
    assert obj["version"] == "v1-elo-pool"