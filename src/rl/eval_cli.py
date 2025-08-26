#!/usr/bin/env python3
"""
Standalone evaluation CLI for RL checkpoints on MPE simple_adversary.
"""

import argparse
import torch
import sys
import os
from pathlib import Path
import math

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl.ppo_selfplay_skeleton import PolicyHead, ValueHead, N_ACT
from src.rl.env_api import make_adapter
from src.rl.models import MultiHeadPolicy, DimAdapter
from src.rl.checkpoint import load_policy_from_ckpt, load_legacy_checkpoint
from src.rl.normalizer import RunningNorm
import src.rl.adapters  # Import to register adapters


def wilson(p, n, z=1.96):
    """Wilson 95% confidence interval for binomial proportion."""
    if n == 0:
        return p, 0.0, 1.0
    z_sq = z * z
    denominator = 1 + z_sq / n
    center = (p + z_sq / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z_sq / (4 * n)) / n) / denominator
    return center, max(0.0, center - margin), min(1.0, center + margin)


def run_episodes(policy, adapter, agents, roles, episodes: int, seed: int, deterministic: bool = True, adapters=None, norms=None):
    """Run episodes using adapter API and return win rate for learner (good) side."""
    import numpy as np
    import torch
    
    wins = 0
    total = 0
    for ep in range(episodes):
        ts = adapter.reset(seed=seed + ep)
        done = False
        # track per-role returns if you need win/loss by comparing sums
        rets = {a: 0.0 for a in agents}
        while not all(ts.dones.values()):
            actions = {}
            for agent in agents:
                role = roles[agent]
                obs_np = ts.obs[agent]
                
                # Apply frozen normalization if available
                if norms and role in norms:
                    obs_np = norms[role].transform(obs_np)
                
                obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
                
                # Apply dimension adapter if needed
                if adapters and role in adapters:
                    obs = adapters[role](obs)
                
                with torch.no_grad():
                    logits = policy.act(role, obs)
                if deterministic:
                    a = int(torch.argmax(logits, dim=-1).item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    a = int(torch.distributions.Categorical(probs=probs).sample().item())
                actions[agent] = a
            ts = adapter.step(actions)
            for a in agents:
                rets[a] += float(ts.rewards[a])
        # decide win/loss for the learner side (assume learner plays "good" unless you toggle)
        good_agents = [a for a in agents if roles[a] == "good"]
        adv_agents = [a for a in agents if roles[a] == "adv"]
        good_total = sum(rets[a] for a in good_agents) if good_agents else 0.0
        adv_total = sum(rets[a] for a in adv_agents) if adv_agents else 0.0
        if good_total > adv_total:
            wins += 1
        total += 1
    return wins / max(1, total)


def main():
    print(f"[cwd] {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='Evaluate RL checkpoint')
    parser.add_argument('--ckpt', help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument("--env", type=str, default="mpe_adversary",
                        help="Environment adapter name (e.g., mpe_adversary, dota_last_hit)")
    parser.add_argument("--list-envs", action="store_true",
                        help="List available env adapters and exit")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--pool', default='artifacts/rl_opponents.json', help='Opponent pool JSON file')
    parser.add_argument('--allow-dim-adapter', action='store_true', default=False,
                       help='Allow dimension adapter for obs dim mismatches')
    parser.add_argument("--dota-difficulty", type=float, default=2.0,
                        help="Difficulty level (0..3) for dota_last_hit eval")
    args = parser.parse_args()
    
    # Handle list-envs option
    from src.rl.env_api import make_adapter, _REGISTRY
    if getattr(args, "list_envs", False):
        print("Available adapters:", ", ".join(sorted(_REGISTRY.keys())))
        raise SystemExit(0)
    
    # Validate required arguments
    if not args.ckpt:
        parser.error("the following arguments are required: --ckpt")
    
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
        
        # Set difficulty for dota_last_hit environment
        if hasattr(adapter, "set_difficulty"):
            adapter.set_difficulty(args.dota_difficulty)
            print(f"Set difficulty to {args.dota_difficulty}")
            
        adapter.reset(seed=args.seed)
        print(f"Created environment: {args.env}")
        
        # Get role mapping and observation dimensions from adapter
        roles = adapter.roles()
        obs_dims = adapter.obs_dims()
        n_actions = adapter.n_actions()
        agents = adapter.agent_names()
        print(f"[roles] {roles}")
        print(f"[obs_dims] {obs_dims}")
        print(f"[n_actions] {n_actions}")
        print(f"[agents] {agents}")
        
    except ValueError as e:
        print(e)
        print("Tip: run with --list-envs to see registered adapters.")
        return 2
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
    norms = {}  # Initialize empty norms dict
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
                    "Run with --allow-dim-adapter to insert a Linear adapter for dimension mismatch."
                )
            else:
                print(f"WARNING: Using dimension adapters for mismatch between ckpt and env dims")
        
        # Use n_actions from adapter or fallback to checkpoint metadata  
        # Build MultiHeadPolicy using checkpoint dimensions
        policy = MultiHeadPolicy(saved_dims, n_actions)
        
        print(f"[checkpoint obs_dims] good={saved_dims['good']}, adv={saved_dims['adv']}")
        print(f"[model expects] good={saved_dims['good']}, adv={saved_dims['adv']}")
        
        # Check checkpoint format and schema
        try:
            sd, meta = load_legacy_checkpoint(ckpt_path)
            schema = meta.get("schema", "unknown")
            print(f"schema: {schema}")
            
            # Validate role structure
            has_roles = any(k.startswith(("pi.good.","pi.adv.","vf.good.","vf.adv.")) for k in sd.keys())
            if not has_roles:
                raise SystemExit("[fatal] legacy checkpoint without role labels. Re-train a new checkpoint after the v2-roles patch.")
            
            # Load checkpoint using robust policy loading
            saved_dims = load_policy_from_ckpt(policy, ckpt, expect_dims=saved_dims)
            print(f"[policy dims] using ckpt dims: good={saved_dims['good']} adv={saved_dims['adv']}")
            
            # Load frozen normalizers if available
            norm_meta = meta.get('norm')
            if norm_meta:
                print("Loading frozen normalizers from checkpoint...")
                for role, norm_state in norm_meta.items():
                    norms[role] = RunningNorm()
                    norms[role].load_state_dict(norm_state)
                    print(f"  {role}: count={norms[role].count:.0f}, mean_norm={abs(norms[role].mean).mean() if norms[role].mean is not None else 0:.3f}")
            else:
                print("No normalizers found in checkpoint metadata")
            
        except Exception as e:
            print(f"Checkpoint loading failed: {e}")
            return 1
        
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
        print(f"Network expects per-role dims: good={saved_dims['good']}, adv={saved_dims['adv']}, n_actions={n_actions}")
        step = ckpt.get("step", 0)
        config = ckpt.get("config", {})
        
        print(f"Loaded checkpoint from step {step}")
        if config:
            print(f"Config: {config}")
            
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 1
    
    # Set networks to evaluation mode
    policy.eval()
    
    # Create seeded RNG for evaluation
    import numpy as np
    rng = np.random.Generator(np.random.PCG64(args.seed))
    
    # Load opponent pool (try new Elo-based format first)
    pool = None
    pool_stats = None
    try:
        from src.rl.opponent_pool import OpponentPool
        pool = OpponentPool.load(args.pool)
        # Check compatibility
        if pool.env == args.env and pool.obs_dims == env_dims:
            entries = pool.list_by_elo()
            min_games = args.opp_min_games if hasattr(args, 'opp_min_games') else 5
            qualified_entries = [e for e in entries if e.games >= min_games]
            pool_stats = {
                'size': len(pool.entries),
                'qualified': len(qualified_entries),
                'elo_mean': sum(e.elo for e in qualified_entries) / max(1, len(qualified_entries)),
                'elo_std': np.std([e.elo for e in qualified_entries]) if len(qualified_entries) > 1 else 0.0
            }
            print(f"Loaded Elo pool: {pool_stats['size']} total, {pool_stats['qualified']} qualified")
            print(f"Pool Elo: mean={pool_stats['elo_mean']:.1f}, std={pool_stats['elo_std']:.1f}")
        else:
            print(f"Pool env/dims mismatch: pool.env={pool.env}, pool.obs_dims={pool.obs_dims}")
            pool = None
    except Exception as e:
        print(f"Could not load Elo pool ({e}), falling back to legacy eval")
        # Fallback to old opponent pool if needed
        try:
            from src.rl.opponent_pool import OpponentPool as LegacyPool
            pool = LegacyPool(args.pool, max_size=1000)  # Large size for eval
        except Exception:
            pool = None
    
    # Run comprehensive evaluation
    try:
        print(f"\nRunning comprehensive evaluation over {args.episodes} episodes...")
        
        results = {}
        
        # 1. Self-play (mirror match)
        print("1/4: Self vs self mirror...")
        wins = args.episodes // 4
        results['wr_self'] = run_episodes(policy, adapter, agents, roles, episodes=wins, seed=args.seed, adapters=adapters, norms=norms)
        center, lower, upper = wilson(results['wr_self'], wins)
        print(f"   Self-play win rate: {results['wr_self']:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
        
        # 2. vs best.pt if exists
        best_path = ckpt_path.parent / "best.pt"
        if best_path.exists() and best_path != ckpt_path:
            print("2/4: vs best.pt...")
            try:
                best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                opp_policy = MultiHeadPolicy(saved_dims, n_actions)
                load_policy_from_ckpt(opp_policy, best_ckpt, expect_dims=saved_dims)
                
                # Temporarily load opponent weights into good role, keep learner as adversary
                original_good_state = policy.pi["good"].state_dict().copy()
                original_good_vf_state = policy.vf["good"].state_dict().copy()
                policy.pi["good"].load_state_dict(opp_policy.pi["good"].state_dict())
                policy.vf["good"].load_state_dict(opp_policy.vf["good"].state_dict())
                
                # Run evaluation (win rate from good perspective, so 1-wr = learner adv win rate)
                episodes_best = args.episodes // 4
                wr = run_episodes(policy, adapter, agents, roles, episodes=episodes_best, seed=args.seed + 1000, adapters=adapters, norms=norms)
                results['wr_best'] = 1.0 - wr  # Learner (adv) wins when good loses
                center, lower, upper = wilson(results['wr_best'], episodes_best)
                print(f"   vs best.pt: learner_wr={results['wr_best']:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
                
                # Restore original weights
                policy.pi["good"].load_state_dict(original_good_state)
                policy.vf["good"].load_state_dict(original_good_vf_state)
            except Exception as e:
                print(f"   vs best.pt failed: {e}")
                results['wr_best'] = 0.5
        else:
            print("2/4: vs best.pt (skipped - not found)")
            results['wr_best'] = 0.5
        
        # 3. vs uniform pool opponents (Elo or legacy)
        if pool and hasattr(pool, 'entries'):  # Elo-based pool
            print("3/4: vs uniform pool opponents (Elo)...")
            uniform_wins = 0
            uniform_games = 0
            qualified_entries = [e for e in pool.entries.values() if e.games >= 5]
            episodes_per_opponent = max(1, (args.episodes // 4) // min(5, len(qualified_entries)))
            
            # Sample 5 opponents uniformly by Elo rank
            import random
            selected_entries = random.sample(qualified_entries, min(5, len(qualified_entries)))
            
            for entry in selected_entries:
                try:
                    opp_ckpt = torch.load(entry.ckpt_path, map_location="cpu", weights_only=False)
                    opp_policy = MultiHeadPolicy(saved_dims, n_actions)
                    load_policy_from_ckpt(opp_policy, opp_ckpt, expect_dims=saved_dims)
                    
                    # Load opponent weights based on its role
                    original_good_state = policy.pi["good"].state_dict().copy()
                    original_good_vf_state = policy.vf["good"].state_dict().copy()
                    if entry.role == "good":
                        policy.pi["good"].load_state_dict(opp_policy.pi["good"].state_dict())
                        policy.vf["good"].load_state_dict(opp_policy.vf["good"].state_dict())
                    
                    wr = run_episodes(policy, adapter, agents, roles, episodes=episodes_per_opponent, seed=args.seed + uniform_games + 2000, adapters=adapters, norms=norms)
                    learner_wr = 1.0 - wr if entry.role == "good" else wr
                    if learner_wr > 0.5:
                        uniform_wins += 1
                    uniform_games += 1
                    center, lower, upper = wilson(learner_wr, episodes_per_opponent)
                    print(f"   vs {entry.id} (elo={entry.elo:.1f}): learner_wr={learner_wr:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
                    
                    # Restore original weights
                    policy.pi["good"].load_state_dict(original_good_state)
                    policy.vf["good"].load_state_dict(original_good_vf_state)
                except Exception as e:
                    print(f"   vs {entry.id}: failed ({e})")
            
            results['wr_pool_uniform'] = uniform_wins / uniform_games if uniform_games > 0 else 0.5
        elif pool:  # Legacy pool
            print("3/4: vs uniform pool opponents (legacy)...")
            # ... keep existing legacy code for uniform sampling
            results['wr_pool_uniform'] = 0.5  # placeholder
        else:
            print("3/4: vs uniform pool (skipped - no pool)")
            results['wr_pool_uniform'] = 0.5
        
        # 4. vs top 5 Elo opponents (or prioritized for legacy)
        if pool and hasattr(pool, 'entries'):  # Elo-based pool
            print("4/4: vs top 5 Elo opponents...")
            top_wins = 0
            top_games = 0
            top_entries = pool.list_by_elo()[:5]  # Top 5 by Elo
            episodes_per_opponent = max(1, (args.episodes // 4) // min(5, len(top_entries)))
            
            for entry in top_entries:
                if entry.games < 5:  # Skip unqualified
                    continue
                try:
                    opp_ckpt = torch.load(entry.ckpt_path, map_location="cpu", weights_only=False)
                    opp_policy = MultiHeadPolicy(saved_dims, n_actions)
                    load_policy_from_ckpt(opp_policy, opp_ckpt, expect_dims=saved_dims)
                    
                    # Load opponent weights based on its role
                    original_good_state = policy.pi["good"].state_dict().copy()
                    original_good_vf_state = policy.vf["good"].state_dict().copy()
                    if entry.role == "good":
                        policy.pi["good"].load_state_dict(opp_policy.pi["good"].state_dict())
                        policy.vf["good"].load_state_dict(opp_policy.vf["good"].state_dict())
                    
                    wr = run_episodes(policy, adapter, agents, roles, episodes=episodes_per_opponent, seed=args.seed + top_games + 3000, adapters=adapters, norms=norms)
                    learner_wr = 1.0 - wr if entry.role == "good" else wr
                    if learner_wr > 0.5:
                        top_wins += 1
                    top_games += 1
                    center, lower, upper = wilson(learner_wr, episodes_per_opponent)
                    print(f"   vs {entry.id} (elo={entry.elo:.1f}, rank #{pool.list_by_elo().index(entry)+1}): learner_wr={learner_wr:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
                    
                    # Restore original weights
                    policy.pi["good"].load_state_dict(original_good_state)
                    policy.vf["good"].load_state_dict(original_good_vf_state)
                except Exception as e:
                    print(f"   vs {entry.id}: failed ({e})")
            
            results['wr_pool_prioritized'] = top_wins / top_games if top_games > 0 else 0.5
            # Also report top-5 specific metric
            results['wr_top5'] = results['wr_pool_prioritized']
        elif pool:  # Legacy pool
            print("4/4: vs prioritized pool opponents (legacy)...")
            # ... keep existing legacy code for prioritized sampling
            results['wr_pool_prioritized'] = 0.5  # placeholder
        else:
            print("4/4: vs prioritized pool (skipped - no pool)")
            results['wr_pool_prioritized'] = 0.5
        
        # Print summary with Wilson CIs
        print(f"\nEvaluation Results Summary:")
        episodes_per_test = args.episodes // 4
        
        center, lower, upper = wilson(results['wr_self'], episodes_per_test)
        print(f"Self-play WR:      {results['wr_self']:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
        
        center, lower, upper = wilson(results['wr_best'], episodes_per_test)
        print(f"vs best.pt WR:     {results['wr_best']:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
        
        # For pool results, estimate episodes based on number of opponents faced
        center, lower, upper = wilson(results['wr_pool_uniform'], episodes_per_test)
        print(f"vs uniform pool:   {results['wr_pool_uniform']:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
        
        center, lower, upper = wilson(results['wr_pool_prioritized'], episodes_per_test)
        print(f"vs priority pool:  {results['wr_pool_prioritized']:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
        
        # Print pool statistics if available
        if pool_stats:
            print(f"\nPool Statistics:")
            print(f"Pool size:         {pool_stats['size']} total, {pool_stats['qualified']} qualified")
            print(f"Pool Elo:          mean={pool_stats['elo_mean']:.1f}, std={pool_stats['elo_std']:.1f}")
            if 'wr_top5' in results:
                print(f"vs top-5 WR:       {results['wr_top5']:.3f}")
        
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