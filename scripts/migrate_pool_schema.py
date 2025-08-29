#!/usr/bin/env python3
"""
Migration script for opponent pool schema: legacy → v1-elo-pool

Usage:
    python scripts/migrate_pool_schema.py --pool-path artifacts/rl_opponents.json [--inplace] [--dry-run]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# add project root (parent of 'scripts') to sys.path so 'src' is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.elo_pool import OpponentPoolV1


def main():
    parser = argparse.ArgumentParser(description='Migrate opponent pool to v1-elo-pool schema')
    parser.add_argument('--pool-path', required=True, help='Path to pool JSON file')
    parser.add_argument('--inplace', action='store_true', help='Overwrite existing file')
    parser.add_argument('--dry-run', action='store_true', help='Show migration without saving')
    
    args = parser.parse_args()
    
    pool_path = Path(args.pool_path)
    
    if not pool_path.exists():
        # Create minimal v1 skeleton
        pool = {
            "version": "v1-elo-pool",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {"elo_k": 32, "scale": 400, "initial": 1200},
            "agents": []
        }
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write using temporary file + os.replace
        import os
        tmp_path = pool_path.with_suffix('.tmp')
        with open(tmp_path, "w") as f:
            json.dump(pool, f, indent=2)
        os.replace(tmp_path, pool_path)
        print("[pool] initialized")
        return 0
    
    # Load existing pool
    with open(pool_path, 'r') as f:
        legacy_data = json.load(f)
    
    # Check if already v1
    if legacy_data.get("version") == "v1-elo-pool":
        print(f"Pool is already v1-elo-pool format: {pool_path}")
        return 0
    
    print(f"[pool] migrate: legacy→v1-elo-pool")
    print(f"Legacy pool: {len(legacy_data.get('agents', []))} agents")
    
    # Migrate to v1
    v1_data = OpponentPoolV1.migrate_legacy(legacy_data)
    
    print(f"v1 pool: {len(v1_data['agents'])} agents")
    print(f"Config: {v1_data['config']}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        print(json.dumps(v1_data, indent=2))
        return 0
    
    # Save migrated pool
    if args.inplace:
        output_path = pool_path
    else:
        output_path = pool_path.with_stem(pool_path.stem + "_v1")
    
    # Atomic write using temporary file + os.replace
    import os
    tmp_path = output_path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(v1_data, f, indent=2)
    os.replace(tmp_path, output_path)
    
    print(f"Migrated pool saved to: {output_path}")
    
    if args.inplace:
        print("[pool] migrated legacy→v1-elo-pool")
    else:
        print("[pool] migrated legacy→v1-elo-pool")
        print(f"To use: mv {output_path} {pool_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())