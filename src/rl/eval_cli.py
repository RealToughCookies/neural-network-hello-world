#!/usr/bin/env python3
"""
Standalone evaluation CLI for RL checkpoints on MPE simple_adversary.
"""

import argparse
import torch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl.ppo_selfplay_skeleton import PolicyHead, ValueHead, make_env, _load_rl_ckpt, N_ACT
from src.rl.selfplay import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL checkpoint')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=4, help='Number of evaluation episodes')
    parser.add_argument('--env', default='mpe_adversary', help='Environment to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    
    print(f"Evaluating checkpoint: {args.ckpt}")
    print(f"Environment: {args.env}, Episodes: {args.episodes}")
    
    # Create environment
    try:
        env = make_env(args.env, seed=args.seed)
        print(f"Created environment: {args.env}")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return 1
    
    # Create role-specific policy and value heads (same architecture as training)
    pi_good = PolicyHead(10, n_act=N_ACT)  # Good agents: 10-D obs
    vf_good = ValueHead(10)
    pi_adv = PolicyHead(8, n_act=N_ACT)   # Adversary agents: 8-D obs  
    vf_adv = ValueHead(8)
    
    # Load checkpoint
    try:
        if not os.path.exists(args.ckpt):
            print(f"Checkpoint not found: {args.ckpt}")
            return 1
            
        ckpt = _load_rl_ckpt(args.ckpt, pi_good, vf_good, pi_adv, vf_adv)
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