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
        
        meta = ckpt.get("meta", {})
        sdg, sda = ckpt.get("pi_good", {}), ckpt.get("pi_adv", {})
        sdg_in, sda_in = _first_layer_in_dim(sdg), _first_layer_in_dim(sda)
        n_act = meta.get("n_act") or _final_layer_out_dim(sdg) or 5
        
        # Fallback to env if needed
        if sdg_in is None or sda_in is None:
            try:
                obs0, _ = env.reset()
            except Exception:
                obs0 = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
            
            keys = list(obs0.keys()) if isinstance(obs0, dict) else []
            gk = next((k for k in keys if "agent" in k), None)
            ak = next((k for k in keys if "adversary" in k), None)
            if sdg_in is None and gk: sdg_in = int(env.observation_space(gk).shape[0])
            if sda_in is None and ak: sda_in = int(env.observation_space(ak).shape[0])
            if n_act is None and (gk or ak):
                anyk = gk or ak; n = getattr(env.action_space(anyk), "n", None); n_act = int(n or 5)
        
        # Preferred dims: good=10, adv=8 (per MPE). If ckpt is reversed, detect and swap.
        good_in, adv_in = 10, 8
        swap = False
        if sdg_in in (8,10) and sda_in in (8,10):
            if sdg_in == 8 and sda_in == 10:
                swap = True
            elif sdg_in == 10 and sda_in == 8:
                swap = False
            else:
                # unexpected, but proceed with sd-inferred dims
                good_in, adv_in = sdg_in, sda_in
        
        print(f"[eval heads] good_in={good_in} adv_in={adv_in} n_act={n_act} swap={swap}")
        # Build heads
        pi_good = PolicyHead(good_in, n_act=n_act)
        vf_good = ValueHead(good_in)
        pi_adv = PolicyHead(adv_in, n_act=n_act)
        vf_adv = ValueHead(adv_in)
        
        # Load with optional swap
        if not swap:
            pi_good.load_state_dict(sdg, strict=True)
            vf_good.load_state_dict(ckpt["vf_good"], strict=True)
            pi_adv.load_state_dict(sda, strict=True)
            vf_adv.load_state_dict(ckpt["vf_adv"], strict=True)
        else:
            print("[eval] ckpt heads appear reversed; swapping pi_good<->pi_adv for load")
            pi_good.load_state_dict(sda, strict=True)
            vf_good.load_state_dict(ckpt["vf_adv"], strict=True)
            pi_adv.load_state_dict(sdg, strict=True)
            vf_adv.load_state_dict(ckpt["vf_good"], strict=True)
        
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