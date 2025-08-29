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
from src.rl.ckpt_io import load_checkpoint_auto
from src.rl.elo_pool import OpponentPoolV1
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
    parser.add_argument('--pool-path', default='artifacts/rl_opponents.json', help='v1 Elo pool JSON file')
    parser.add_argument('--allow-dim-adapter', action='store_true', default=False,
                       help='Allow dimension adapter for obs dim mismatches')
    parser.add_argument("--dota-difficulty", type=float, default=2.0,
                        help="Difficulty level (0..3) for dota_last_hit eval")
    parser.add_argument(
        "--opp-sample",
        choices=["uniform", "topk", "pfsp_elo"],
        default="pfsp_elo",
        help="Opponent sampling strategy for pool buckets."
    )
    parser.add_argument(
        "--opp-topk", type=int, default=5,
        help="K for top-k sampling when --opp-sample=topk (or for a top-K bucket)."
    )
    parser.add_argument(
        "--opp-tau", type=float, default=1.5,
        help="Tau for PFSP-elo weighting (higher = softer focus)."
    )
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
    ckpt_arg = Path(args.ckpt)
    if ckpt_arg.is_dir():
        # Try in order
        for name in ("last.pt", "last_model.pt", "best.pt"):
            candidate = ckpt_arg / name
            if candidate.exists():
                print(f"Directory provided, auto-selected: {candidate.name}")
                ckpt_arg = candidate
                break
        else:
            raise FileNotFoundError(f"No checkpoint files found in {ckpt_arg}; looked for last.pt, last_model.pt, best.pt")
    
    ckpt_path = ckpt_arg
    
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
    adapter = None  # single adapter instance
    try:
        adapter = make_adapter(args.env, render_mode=None)
        
        # Set difficulty for dota_last_hit environment
        if hasattr(adapter, "set_difficulty"):
            adapter.set_difficulty(args.dota_difficulty)
            print(f"Set difficulty to {args.dota_difficulty}")
        
        # Define roles once - use adapter.agents if available
        roles = adapter.roles()  # dict mapping agent_name -> role
        agents = getattr(adapter, "agents", None) or adapter.agent_names()
        assert isinstance(agents, (list, tuple)) and agents, f"Adapter must expose agent list; got {type(agents)}"
        
        # Initialize dimension adapters dict (for obs dimension mismatches)
        adapters = {}
            
        adapter.reset(seed=args.seed)
        print(f"Created environment: {args.env}")
        
        # Get observation dimensions and actions from adapter
        obs_dims = adapter.obs_dims()
        n_actions = adapter.n_actions()
        print(f"[roles] {roles}")
        print(f"[agents] {agents}")
        print(f"[obs_dims] {obs_dims}")
        print(f"[n_actions] {n_actions}")
        
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
    
    # Load checkpoint using helper  
    device = "cpu"
    obs_normalizers = {}
    policy = MultiHeadPolicy(obs_dims, n_actions)
    try:
        kind, payload = load_checkpoint_auto(Path(args.ckpt), map_location=device)
        if kind == "v3":
            b = payload
            policy.load_state_dict(b["model_state"])
            step = b.get("counters", {}).get("global_step", 0)
            seed = b.get("meta", {}).get("seed")
            print(f"[checkpoint kind] v3 | step={step} seed={seed}")
            if b.get("obs_norm_state"):
                for r, sd in b["obs_norm_state"].items():
                    obs_normalizers[r] = RunningNorm()
                    obs_normalizers[r].load_state_dict(sd)
                print("Applied frozen obs-normalizers from checkpoint")
            ckpt = {"model": b["model_state"], "meta": b["meta"]}
        else:
            policy.load_state_dict(payload)
            print("[checkpoint kind] model_only (no normalizers)")
            ckpt = payload
        
        saved_dims = ckpt.get("meta", {}).get("obs_dims") if isinstance(ckpt, dict) else None
        if saved_dims is None:
            raise ValueError("Checkpoint missing meta.obs_dims")
        
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
        
        # Add "tripwire" assertions where it matters
        def _assert_is_mapping(name, obj):
            import collections.abc
            assert isinstance(obj, collections.abc.Mapping), f"{name} must be a dict-like mapping, got {type(obj)}"
        
        _assert_is_mapping("obs_normalizers", obs_normalizers)
        assert isinstance(obs_normalizers, dict), f"obs_normalizers is {type(obs_normalizers)}; did you forget to call the factory?"
        
        # Use obs_normalizers as norms for compatibility
        norms = obs_normalizers
            
    except Exception as e:
        print(f"Checkpoint loading failed: {e}")
        return 1
        
        # Add adapters if environment dimensions don't match checkpoint
        adapters = {}
        if env_dims != saved_dims and args.allow_dim_adapter:
            for role in roles:
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
    
    # Load v1 Elo pool if provided
    def load_pool_if_any(pool_path):
        if hasattr(args, 'pool_path') and Path(pool_path).exists():
            try:
                pool = OpponentPoolV1(Path(pool_path))
                pool.load()
                return pool
            except Exception as e:
                print(f"Could not load v1 pool: {e}")
                return None
        return None

    pool = load_pool_if_any(args.pool_path)  # your v1 loader
    def num_agents(p):
        if p is None: return 0
        if hasattr(p, "agents"): return len(p.agents)
        if isinstance(p, dict):  return len(p.get("agents", []))
        return 0

    pool_stats = None
    n_agents = len(pool.data["agents"]) if (pool and hasattr(pool, "data") and "agents" in pool.data) else 0
    if n_agents == 0:
        print("[pool] loaded v1-elo-pool (agents=0)")
        print("Skipping pool buckets (no agents).")
        run_pool_buckets = False
    else:
        run_pool_buckets = True
        # ... run top-K/uniform/PFSP
        pool_agents = pool.data["agents"]
        pool_stats = {
            'size': len(pool_agents),
            'qualified': len(pool_agents), # all agents are qualified
            'elo_mean': sum(a["elo"] for a in pool_agents) / len(pool_agents),
            'elo_std': np.std([a["elo"] for a in pool_agents]) if len(pool_agents) > 1 else 0.0
        }
        print(f"[pool] loaded v1-elo-pool (agents={pool_stats['size']})")
        print(f"Pool Elo: mean={pool_stats['elo_mean']:.1f}, std={pool_stats['elo_std']:.1f}")
    
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
        
        # 3. vs v1 pool opponents - three buckets evaluation
        if run_pool_buckets:
            episodes_per_bucket = args.episodes // 6  # Split remaining episodes across 3 buckets
            
            # Bucket 1: Top-K (K = min(args.opp_topk, n_agents))
            K = min(args.opp_topk, n_agents)
            print(f"3/6: vs top-{K} Elo...")
            topk_wins = 0
            topk_games = 0
            for i in range(K):
                try:
                    agent = pool.sample(strategy="topk", topk=K)
                    kind, obj = load_checkpoint_auto(Path(agent["path"]))
                    opp_policy = MultiHeadPolicy(saved_dims, n_actions)
                    if kind == "v3":
                        ckpt = {"model": obj["model_state"], "meta": obj["meta"]}
                    else:
                        ckpt = obj
                    load_policy_from_ckpt(opp_policy, ckpt, expect_dims=saved_dims)
                    
                    # Temp opponent swap (assume learner=adv, opponent=good)
                    original_good_state = policy.pi["good"].state_dict().copy()
                    original_good_vf_state = policy.vf["good"].state_dict().copy()
                    policy.pi["good"].load_state_dict(opp_policy.pi["good"].state_dict())
                    policy.vf["good"].load_state_dict(opp_policy.vf["good"].state_dict())
                    
                    wr = run_episodes(policy, adapter, agents, roles, episodes=episodes_per_bucket, seed=args.seed + i + 2000, adapters=adapters, norms=norms)
                    learner_wr = 1.0 - wr  # Learner is adv, so win when good loses
                    if learner_wr > 0.5:
                        topk_wins += 1
                    topk_games += 1
                    
                    # Restore
                    policy.pi["good"].load_state_dict(original_good_state)
                    policy.vf["good"].load_state_dict(original_good_vf_state)
                    
                    center, lower, upper = wilson(learner_wr, episodes_per_bucket)
                    print(f"   vs {agent['id']} (elo={agent['elo']:.1f}): wr={learner_wr:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
                except Exception as e:
                    print(f"   topk failed: {e}")
            
            results['wr_topk'] = topk_wins / topk_games if topk_games > 0 else 0.5
            
            # Bucket 2: Uniform 
            print(f"4/6: vs uniform (N={K})...")
            uniform_wins = 0
            uniform_games = 0
            for i in range(K):  # Sample K uniform opponents
                try:
                    agent = pool.sample(strategy="uniform")
                    kind, obj = load_checkpoint_auto(Path(agent["path"]))
                    opp_policy = MultiHeadPolicy(saved_dims, n_actions)
                    if kind == "v3":
                        ckpt = {"model": obj["model_state"], "meta": obj["meta"]}
                    else:
                        ckpt = obj
                    load_policy_from_ckpt(opp_policy, ckpt, expect_dims=saved_dims)
                    
                    # Temp opponent swap (assume learner=adv, opponent=good)
                    original_good_state = policy.pi["good"].state_dict().copy()
                    original_good_vf_state = policy.vf["good"].state_dict().copy()
                    policy.pi["good"].load_state_dict(opp_policy.pi["good"].state_dict())
                    policy.vf["good"].load_state_dict(opp_policy.vf["good"].state_dict())
                    
                    wr = run_episodes(policy, adapter, agents, roles, episodes=episodes_per_bucket, seed=args.seed + i + 3000, adapters=adapters, norms=norms)
                    learner_wr = 1.0 - wr  # Learner is adv, so win when good loses
                    if learner_wr > 0.5:
                        uniform_wins += 1
                    uniform_games += 1
                    
                    # Restore
                    policy.pi["good"].load_state_dict(original_good_state)
                    policy.vf["good"].load_state_dict(original_good_vf_state)
                    
                    center, lower, upper = wilson(learner_wr, episodes_per_bucket)
                    print(f"   vs {agent['id']} (elo={agent['elo']:.1f}): wr={learner_wr:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
                except Exception as e:
                    print(f"   uniform failed: {e}")
            results['wr_pool_uniform'] = uniform_wins / uniform_games if uniform_games > 0 else 0.5
            
            # Bucket 3: PFSP-Elo
            print(f"5/6: vs pfsp_elo (tau={args.opp_tau})...")
            pfsp_wins = 0
            pfsp_games = 0
            for i in range(K):  # Sample K PFSP opponents
                try:
                    agent = pool.sample(strategy="pfsp_elo", tau=args.opp_tau)
                    kind, obj = load_checkpoint_auto(Path(agent["path"]))
                    opp_policy = MultiHeadPolicy(saved_dims, n_actions)
                    if kind == "v3":
                        ckpt = {"model": obj["model_state"], "meta": obj["meta"]}
                    else:
                        ckpt = obj
                    load_policy_from_ckpt(opp_policy, ckpt, expect_dims=saved_dims)
                    
                    # Temp opponent swap (assume learner=adv, opponent=good)
                    original_good_state = policy.pi["good"].state_dict().copy()
                    original_good_vf_state = policy.vf["good"].state_dict().copy()
                    policy.pi["good"].load_state_dict(opp_policy.pi["good"].state_dict())
                    policy.vf["good"].load_state_dict(opp_policy.vf["good"].state_dict())
                    
                    wr = run_episodes(policy, adapter, agents, roles, episodes=episodes_per_bucket, seed=args.seed + i + 4000, adapters=adapters, norms=norms)
                    learner_wr = 1.0 - wr  # Learner is adv, so win when good loses
                    if learner_wr > 0.5:
                        pfsp_wins += 1
                    pfsp_games += 1
                    
                    # Restore
                    policy.pi["good"].load_state_dict(original_good_state)
                    policy.vf["good"].load_state_dict(original_good_vf_state)
                    
                    center, lower, upper = wilson(learner_wr, episodes_per_bucket)
                    print(f"   vs {agent['id']} (elo={agent['elo']:.1f}): wr={learner_wr:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")
                except Exception as e:
                    print(f"   pfsp failed: {e}")
            results['wr_pool_prioritized'] = pfsp_wins / pfsp_games if pfsp_games > 0 else 0.5
            
        else:
            print("3/6: vs pool (skipped - no pool)")
            results['wr_topk'] = 0.5
            results['wr_pool_uniform'] = 0.5
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
    raise SystemExit(main())