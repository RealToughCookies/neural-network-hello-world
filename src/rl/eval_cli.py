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

from src.rl.ppo_selfplay_skeleton import PolicyHead, ValueHead, make_env, _load_rl_ckpt, N_ACT
from src.rl.selfplay import evaluate


def main():
    print(f"[cwd] {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description='Evaluate RL checkpoint')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=4, help='Number of evaluation episodes')
    parser.add_argument('--env', default='mpe_adversary', help='Environment to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
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
    
    # Create environment
    try:
        env = make_env(args.env, seed=args.seed)
        print(f"Created environment: {args.env}")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return 1
    
    # Load checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # trusted local file
        
        def _infer_dims_from_ckpt(ckpt):
            """Infer network dimensions from checkpoint metadata or weights."""
            meta = ckpt.get("meta", {})
            if all(k in meta for k in ("good_in", "adv_in", "n_act")):
                return meta["good_in"], meta["adv_in"], meta["n_act"]
            
            sdg, sda = ckpt.get("pi_good", {}), ckpt.get("pi_adv", {})
            def _in_dim(sd):  # first Linear weight is [out, in]
                for k, v in sd.items():
                    if k.endswith("net.0.weight"):  # our PolicyHead first layer
                        return int(v.shape[1])
                return None
            
            good_in = _in_dim(sdg)
            adv_in = _in_dim(sda)
            n_act = None
            for k, v in sdg.items():
                if k.endswith("net.2.weight"):  # final layer [n_act, hidden]
                    n_act = int(v.shape[0])
                    break
            return good_in, adv_in, n_act
        
        good_in, adv_in, n_act = _infer_dims_from_ckpt(ckpt)
        
        # Fallback to env if any missing
        if None in (good_in, adv_in, n_act):
            try:
                obs0, _ = env.reset()
            except Exception:
                obs0 = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
            
            # Parallel dict; pick a good agent and adversary key
            keys = list(obs0.keys()) if isinstance(obs0, dict) else []
            good_keys = [k for k in keys if "agent" in k]
            adv_keys = [k for k in keys if "adversary" in k]
            
            if (good_in is None) and good_keys:
                good_in = int(env.observation_space(good_keys[0]).shape[0])
            if (adv_in is None) and adv_keys:
                adv_in = int(env.observation_space(adv_keys[0]).shape[0])
            if n_act is None:
                any_key = (good_keys or adv_keys)[0]
                n_act = int(getattr(env.action_space(any_key), "n", 5))
        
        assert good_in and adv_in and n_act, "Failed to infer head dims/n_act"
        
        # Build heads with correct shapes
        pi_good = PolicyHead(good_in, n_act=n_act)
        vf_good = ValueHead(good_in)
        pi_adv = PolicyHead(adv_in, n_act=n_act)
        vf_adv = ValueHead(adv_in)
        
        print(f"[eval heads] good_in={good_in} adv_in={adv_in} n_act={n_act}")
        
        # Load state dicts into networks
        pi_good.load_state_dict(ckpt["pi_good"], strict=True)
        vf_good.load_state_dict(ckpt["vf_good"], strict=True)
        pi_adv.load_state_dict(ckpt["pi_adv"], strict=True)
        vf_adv.load_state_dict(ckpt["vf_adv"], strict=True)
        
        step = ckpt.get("step", 0)
        config = ckpt.get("config", {})
        
        print(f"Loaded checkpoint from step {step}")
        if config:
            print(f"Config: {config}")
            
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return 1
    
    # Set networks to evaluation mode
    pi_good.eval()
    vf_good.eval() 
    pi_adv.eval()
    vf_adv.eval()
    
    # Run evaluation
    try:
        print(f"\nRunning evaluation over {args.episodes} episodes...")
        mean_good, mean_adv = evaluate(env, pi_good, pi_adv, episodes=args.episodes)
        
        print(f"\nEvaluation Results:")
        print(f"Mean return (good agents): {mean_good:.3f}")
        print(f"Mean return (adversary):   {mean_adv:.3f}")
        print(f"Combined score:           {mean_good + mean_adv:.3f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())