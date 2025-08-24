#!/usr/bin/env python3
"""
Standalone evaluation CLI for RL checkpoints on MPE simple_adversary.
"""

import argparse
import torch
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl.ppo_selfplay_skeleton import PolicyHead, ValueHead, N_ACT
from src.rl.selfplay import evaluate
from src.rl.env_api import make_adapter
from src.rl.models import MultiHeadPolicy, DimAdapter
from src.rl.checkpoint import load_policy_from_ckpt
import src.rl.adapters  # Import to register adapters


def main():
    print(f"[cwd] {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='Evaluate RL checkpoint')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--env', default='mpe_adversary', help='Environment to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--pool', default='artifacts/rl_opponents.json', help='Opponent pool JSON file')
    parser.add_argument('--allow-dim-adapter', action='store_true', default=False,
                       help='Allow dimension adapter for obs dim mismatches')
    args = parser.parse_args()
    
    # Resolve checkpoint path
    ckpt_path = Path(args.ckpt).expanduser().resolve(strict=False)
    print(f"Evaluating checkpoint: {ckpt_path}")
    print(f"Environment: {args.env}, Episodes: {args.episodes}")
    
    if not ckpt_path.exists():
        parent = ckpt_path.parent
        if parent.exists():
            listing = ", ".join(sorted(p.name for p in parent.glob("*.pt")))
            if not listing:
                listing = "<no .pt files>"
        else:
            listing = "<missing dir>"
        raise SystemExit(f"Checkpoint not found: {ckpt_path}\nDir contents: {listing}")
    
    # Create environment adapter
    try:
        adapter = make_adapter(args.env, render_mode=None)
        adapter.reset(seed=args.seed)
        print(f"Created environment: {args.env}")
        
        # Get role mapping and observation dimensions from adapter
        role_of = adapter.roles()
        obs_dims = adapter.obs_dims()
        print(f"[roles] {role_of}")
        print(f"[obs_dims] {obs_dims}")
        
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return 1
    
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
    
    # Load checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # trusted local file
        
        # STRICT: Get dimensions from checkpoint metadata only
        saved_dims = ckpt.get("meta", {}).get("obs_dims")
        if saved_dims is None:
            raise ValueError(
                "Checkpoint missing meta.obs_dims; re-train a small ckpt with the new trainer. "
                "This checkpoint was created before per-role dimension support was added."
            )
        
        print(f"[checkpoint obs_dims] good={saved_dims['good']}, adv={saved_dims['adv']}")
        
        # Compare saved dimensions vs current adapter dimensions
        env_dims = adapter.obs_dims()
        print(f"[env obs_dims] good={env_dims['good']}, adv={env_dims['adv']}")
        
        if env_dims != saved_dims:
            if not args.allow_dim_adapter:
                raise ValueError(
                    f"Dimension mismatch: ckpt={saved_dims}, env={env_dims}. "
                    "Pin pettingzoo<1.25 or run with --allow-dim-adapter to insert a Linear adapter."
                )
            else:
                print(f"WARNING: Using dimension adapters for mismatch between ckpt and env dims")
        
        n_act = ckpt.get("meta", {}).get("n_act") or N_ACT
        
        # Build MultiHeadPolicy using checkpoint dimensions
        policy = MultiHeadPolicy(saved_dims, n_act)
        
        print(f"[checkpoint obs_dims] good={saved_dims['good']}, adv={saved_dims['adv']}")
        print(f"[model expects] good={saved_dims['good']}, adv={saved_dims['adv']}")
        
        # Load checkpoint using robust policy loading
        saved_dims = load_policy_from_ckpt(policy, ckpt, expect_dims=saved_dims)
        print(f"[policy dims] using ckpt dims: good={saved_dims['good']} adv={saved_dims['adv']}")
        
        # Add adapters if environment dimensions don't match checkpoint
        adapters = {}
        if env_dims != saved_dims and args.allow_dim_adapter:
            for role in ["good", "adv"]:
                env_dim = env_dims[role]
                ckpt_dim = saved_dims[role]
                if env_dim != ckpt_dim:
                    adapters[role] = DimAdapter(env_dim, ckpt_dim) 
                    print(f"WARNING: Added adapter for role '{role}': env_dim={env_dim} -> ckpt_dim={ckpt_dim}")
        
        # Print final dimensions the network expects
        print(f"Network expects per-role dims: good={saved_dims['good']}, adv={saved_dims['adv']}, n_act={n_act}")
        step = ckpt.get("step", 0)
        config = ckpt.get("config", {})
        
        print(f"Loaded checkpoint from step {step}")
        if config:
            print(f"Config: {config}")
            
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 1
    
    # Set networks to evaluation mode and extract policy/value heads
    if 'policy' in locals():
        # New format - use MultiHeadPolicy
        policy.eval()
        pi_good = policy.pi["good"]
        pi_adv = policy.pi["adv"] 
        vf_good = policy.vf["good"]
        vf_adv = policy.vf["adv"]
    else:
        # Legacy format - individual heads
        pi_good.eval()
        vf_good.eval() 
        pi_adv.eval()
        vf_adv.eval()
    
    # Create seeded RNG for evaluation
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(args.seed))
    
    # Load opponent pool
    from src.rl.opponent_pool import OpponentPool
    pool = OpponentPool(args.pool, max_size=1000)  # Large size for eval
    
    # Run comprehensive evaluation
    try:
        print(f"\nRunning comprehensive evaluation over {args.episodes} episodes...")
        
        results = {}
        
        # 1. Self-play (mirror match)
        print("1/4: Self vs self mirror...")
        mean_good_self, mean_adv_self = evaluate(adapter.env, pi_good, pi_adv, episodes=args.episodes // 4)
        results['wr_self'] = 0.5  # Always 50% in self-play
        print(f"   Self-play: good={mean_good_self:.3f}, adv={mean_adv_self:.3f}")
        
        # 2. vs best.pt if exists
        best_path = ckpt_path.parent / "best.pt"
        if best_path.exists() and best_path != ckpt_path:
            print("2/4: vs best.pt...")
            try:
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                opp_policy = MultiHeadPolicy(saved_dims, n_act)
                load_policy_from_ckpt(opp_policy, best_ckpt, expect_dims=saved_dims)
                opp_pi_good = opp_policy.pi["good"]
                opp_pi_adv = opp_policy.pi["adv"]
                
                # Evaluate with opponent as good agents, learner as adversary
                _, mean_learner_vs_best = evaluate(adapter.env, opp_pi_good, pi_adv, episodes=args.episodes // 4)
                results['wr_best'] = 1.0 if mean_learner_vs_best > 0 else 0.0  # Win if positive reward
                print(f"   vs best.pt: learner_reward={mean_learner_vs_best:.3f}")
            except Exception as e:
                print(f"   vs best.pt failed: {e}")
                results['wr_best'] = 0.5
        else:
            print("2/4: vs best.pt (skipped - not found)")
            results['wr_best'] = 0.5
        
        # 3. vs 5 uniform pool opponents
        if pool:
            print("3/4: vs 5 uniform pool opponents...")
            uniform_wins = 0
            uniform_games = 0
            episodes_per_opponent = max(1, (args.episodes // 4) // min(5, len(pool)))
            
            for i in range(min(5, len(pool))):
                opponent_ckpt = pool.sample_uniform(rng)
                if opponent_ckpt and Path(opponent_ckpt).exists():
                    try:
                        opp_ckpt = torch.load(opponent_ckpt, map_location="cpu", weights_only=False)
                        opp_policy = MultiHeadPolicy(saved_dims, n_act)
                        load_policy_from_ckpt(opp_policy, opp_ckpt, expect_dims=saved_dims)
                        opp_pi_good = opp_policy.pi["good"]
                        opp_pi_adv = opp_policy.pi["adv"]
                        
                        _, learner_reward = evaluate(adapter.env, opp_pi_good, pi_adv, episodes=episodes_per_opponent)
                        if learner_reward > 0:
                            uniform_wins += 1
                        uniform_games += 1
                        print(f"   vs {Path(opponent_ckpt).name}: reward={learner_reward:.3f}")
                    except Exception as e:
                        print(f"   vs {opponent_ckpt}: failed ({e})")
            
            results['wr_pool_uniform'] = uniform_wins / uniform_games if uniform_games > 0 else 0.5
        else:
            print("3/4: vs uniform pool (skipped - empty pool)")
            results['wr_pool_uniform'] = 0.5
        
        # 4. vs 5 prioritized pool opponents
        if pool:
            print("4/4: vs 5 prioritized pool opponents...")
            prioritized_wins = 0
            prioritized_games = 0
            episodes_per_opponent = max(1, (args.episodes // 4) // min(5, len(pool)))
            
            for i in range(min(5, len(pool))):
                opponent_ckpt = pool.sample_prioritized(rng, temp=0.7)
                if opponent_ckpt and Path(opponent_ckpt).exists():
                    try:
                        opp_ckpt = torch.load(opponent_ckpt, map_location="cpu", weights_only=False)
                        opp_policy = MultiHeadPolicy(saved_dims, n_act)
                        load_policy_from_ckpt(opp_policy, opp_ckpt, expect_dims=saved_dims)
                        opp_pi_good = opp_policy.pi["good"]
                        opp_pi_adv = opp_policy.pi["adv"]
                        
                        _, learner_reward = evaluate(adapter.env, opp_pi_good, pi_adv, episodes=episodes_per_opponent)
                        if learner_reward > 0:
                            prioritized_wins += 1
                        prioritized_games += 1
                        print(f"   vs {Path(opponent_ckpt).name}: reward={learner_reward:.3f}")
                    except Exception as e:
                        print(f"   vs {opponent_ckpt}: failed ({e})")
            
            results['wr_pool_prioritized'] = prioritized_wins / prioritized_games if prioritized_games > 0 else 0.5
        else:
            print("4/4: vs prioritized pool (skipped - empty pool)")
            results['wr_pool_prioritized'] = 0.5
        
        # Print summary
        print(f"\nEvaluation Results Summary:")
        print(f"Self-play WR:      {results['wr_self']:.3f}")
        print(f"vs best.pt WR:     {results['wr_best']:.3f}")
        print(f"vs uniform pool:   {results['wr_pool_uniform']:.3f}")
        print(f"vs priority pool:  {results['wr_pool_prioritized']:.3f}")
        
        # Append to CSV
        csv_path = Path("artifacts/rl_metrics.csv")
        if csv_path.exists():
            import csv
            # Check if we need to add header for new columns
            needs_header = False
            with open(csv_path, 'r') as rf:
                first_line = rf.readline().strip()
                if 'wr_self' not in first_line:
                    needs_header = True
            
            if needs_header:
                # Read existing and add new columns
                with open(csv_path, 'r') as rf:
                    reader = csv.reader(rf)
                    rows = list(reader)
                
                if rows:
                    rows[0].extend(['wr_self', 'wr_best', 'wr_pool_uniform', 'wr_pool_prioritized'])
                    
                    with open(csv_path, 'w', newline='') as wf:
                        writer = csv.writer(wf)
                        writer.writerows(rows)
            
            # Append evaluation results
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [step, results['wr_self'], results['wr_best'], 
                       results['wr_pool_uniform'], results['wr_pool_prioritized']]
                writer.writerow(row)
                print(f"Appended results to {csv_path}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())