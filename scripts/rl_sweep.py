#!/usr/bin/env python3
"""
RL parameter sweep runner for multi-seed experiments.

Runs multiple seeds of PPO training and aggregates results with statistical analysis.
Based on Stable Baselines3 documentation recommendation to run multiple seeds for
robust evaluation of RL algorithms.

References:
- SB3 Tips: https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
- Wilson CI: https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Wilson confidence interval for win rate
def wilson_confidence_interval(wins: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Calculate Wilson score confidence interval for binomial proportion.
    
    Args:
        wins: Number of successes
        total: Total number of trials
        alpha: Significance level (0.05 for 95% CI)
    
    Returns:
        (lower_bound, upper_bound) tuple
    
    Reference:
        https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
    """
    if total == 0:
        return 0.0, 0.0
    
    from math import sqrt
    z = 1.96  # 95% confidence (alpha=0.05)
    p_hat = wins / total
    
    denominator = 1 + (z * z) / total
    center = p_hat + (z * z) / (2 * total)
    margin = z * sqrt((p_hat * (1 - p_hat) + (z * z) / (4 * total)) / total)
    
    lower = (center - margin) / denominator
    upper = (center + margin) / denominator
    
    return max(0.0, lower), min(1.0, upper)


def run_single_experiment(args: argparse.Namespace, seed: int, output_dir: Path) -> Dict[str, Any]:
    """Run a single experiment with given seed."""
    print(f"🎯 Running seed {seed}...")
    
    # Build command for PPO training
    cmd = [
        sys.executable, "-m", "src.rl.ppo_selfplay_skeleton",
        "--env", args.env,
        "--global-seed", str(seed),
        "--updates", str(args.updates),
        "--save-dir", str(output_dir / f"seed_{seed}"),
    ]
    
    # Add parameter overrides
    for param, value in args.param:
        cmd.extend([f"--{param}", str(value)])
    
    # Add training flag
    if args.mode == "selfplay":
        cmd.append("--selfplay")
    else:
        cmd.append("--train")
    
    if args.deterministic:
        cmd.append("--deterministic")
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        success = True
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        success = False
        stdout = e.stdout
        stderr = e.stderr
    
    elapsed = time.time() - start_time
    
    # Try to parse metrics from output (basic parsing)
    final_reward = None
    win_rate = None
    
    # Look for final metrics in stdout
    for line in stdout.split('\n'):
        if 'final_reward' in line.lower():
            try:
                final_reward = float(line.split(':')[-1].strip())
            except ValueError:
                pass
        if 'win_rate' in line.lower() or 'win-rate' in line.lower():
            try:
                win_rate = float(line.split(':')[-1].strip())
            except ValueError:
                pass
    
    # Fallback: extract from manifest if available
    manifest_path = output_dir / f"seed_{seed}" / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception:
            pass
    
    return {
        "seed": seed,
        "success": success,
        "elapsed": elapsed,
        "final_reward": final_reward,
        "win_rate": win_rate,
        "stdout": stdout,
        "stderr": stderr,
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results across seeds with statistical analysis."""
    successful = [r for r in results if r["success"]]
    n_success = len(successful)
    n_total = len(results)
    
    if n_success == 0:
        return {
            "summary": f"All {n_total} runs failed",
            "success_rate": 0.0,
            "rewards": None,
            "win_rates": None,
        }
    
    # Aggregate rewards
    rewards = [r["final_reward"] for r in successful if r["final_reward"] is not None]
    reward_stats = None
    if rewards:
        rewards = np.array(rewards)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0
        
        # 95% confidence interval for mean (t-distribution)
        if len(rewards) > 1:
            from scipy.stats import t
            sem = reward_std / np.sqrt(len(rewards))
            ci_margin = t.ppf(0.975, len(rewards) - 1) * sem
        else:
            ci_margin = 0.0
        
        reward_stats = {
            "mean": float(reward_mean),
            "std": float(reward_std),
            "ci_lower": float(reward_mean - ci_margin),
            "ci_upper": float(reward_mean + ci_margin),
            "n": len(rewards),
        }
    
    # Aggregate win rates
    win_rates = [r["win_rate"] for r in successful if r["win_rate"] is not None]
    win_rate_stats = None
    if win_rates:
        # Assume win rates are proportions; estimate Wilson CI for average
        avg_win_rate = np.mean(win_rates)
        
        # For Wilson CI, we need wins/total, but we have proportions
        # Use approximate method: treat each run as a single trial
        total_wins = sum(win_rates)  # sum of proportions
        total_trials = len(win_rates)
        wilson_lower, wilson_upper = wilson_confidence_interval(int(total_wins), total_trials)
        
        win_rate_stats = {
            "mean": float(avg_win_rate),
            "std": float(np.std(win_rates, ddof=1)) if len(win_rates) > 1 else 0.0,
            "wilson_ci_lower": wilson_lower,
            "wilson_ci_upper": wilson_upper,
            "n": len(win_rates),
        }
    
    return {
        "summary": f"{n_success}/{n_total} runs successful",
        "success_rate": n_success / n_total,
        "rewards": reward_stats,
        "win_rates": win_rate_stats,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RL parameter sweep with multiple seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--env",
        default="mpe_adversary",
        help="Environment name"
    )
    
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of random seeds to run"
    )
    
    parser.add_argument(
        "--updates",
        type=int,
        default=10,
        help="Number of PPO updates per run"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "selfplay"],
        default="train",
        help="Training mode"
    )
    
    parser.add_argument(
        "--param",
        action="append",
        nargs=2,
        metavar=("PARAM", "VALUE"),
        default=[],
        help="Parameter override (e.g., --param clip-range 0.1)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="artifacts/rl_sweep",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithms"
    )
    
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1337,
        help="Starting seed value"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🧪 Starting RL parameter sweep")
    print(f"📁 Output: {output_dir}")
    print(f"🎲 Seeds: {args.seeds} (starting from {args.start_seed})")
    print(f"🎮 Environment: {args.env}")
    print(f"📈 Updates per run: {args.updates}")
    print(f"🎯 Mode: {args.mode}")
    
    if args.param:
        print("⚙️  Parameter overrides:")
        for param, value in args.param:
            print(f"    --{param} {value}")
    
    print("-" * 50)
    
    # Run experiments
    results = []
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    
    for seed in seeds:
        result = run_single_experiment(args, seed, output_dir)
        results.append(result)
        
        status = "✅" if result["success"] else "❌"
        print(f"{status} Seed {seed}: {result['elapsed']:.1f}s")
        if not result["success"]:
            print(f"   Error: {result['stderr'].split(chr(10))[0][:100]}")
    
    print("-" * 50)
    
    # Aggregate and display results
    aggregated = aggregate_results(results)
    print(f"📊 {aggregated['summary']}")
    
    if aggregated["rewards"]:
        r = aggregated["rewards"]
        print(f"🎁 Reward: {r['mean']:.3f} ± {r['std']:.3f} "
              f"[95% CI: {r['ci_lower']:.3f}, {r['ci_upper']:.3f}] (n={r['n']})")
    
    if aggregated["win_rates"]:
        w = aggregated["win_rates"]
        print(f"🏆 Win Rate: {w['mean']:.3f} ± {w['std']:.3f} "
              f"[Wilson CI: {w['wilson_ci_lower']:.3f}, {w['wilson_ci_upper']:.3f}] (n={w['n']})")
    
    # Save results
    results_file = output_dir / "sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "results": results,
            "aggregated": aggregated,
        }, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    return 0 if aggregated["success_rate"] > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(1)
    except ImportError as e:
        if "scipy" in str(e):
            print("📦 Installing scipy for statistical analysis...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
            print("🔄 Please run again")
            sys.exit(0)
        else:
            raise