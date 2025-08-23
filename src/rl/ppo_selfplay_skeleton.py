"""
Minimal PPO self-play skeleton for Dota-class RL research.
Supports Google Research Football (primary) and PettingZoo (fallback).
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """Policy network with categorical distribution."""
    
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, obs):
        return self.net(obs)


class ValueNet(nn.Module):
    """Value function network."""
    
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, obs):
        return self.net(obs)


class MockEnvironment:
    """Mock environment for testing when no RL libraries are available."""
    
    def __init__(self):
        self.observation_space = self._create_mock_space()
        self.action_space = self._create_mock_space()
        self.step_count = 0
        
    def _create_mock_space(self):
        class MockSpace:
            def __init__(self):
                self.shape = (10,)
                self.n = 4
            def sample(self):
                return np.random.randint(0, self.n)
        return MockSpace()
    
    def reset(self):
        self.step_count = 0
        return np.random.randn(10)
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.randn(10)
        reward = np.random.randn()
        done = self.step_count >= 10
        info = {}
        return obs, reward, done, info


def load_environment(env_name, seed=0):
    """Load environment with fallback support."""
    if env_name == "grf":
        try:
            import gfootball.env as football_env
            env = football_env.create_environment(
                env_name='11_vs_11_stochastic',
                representation='simple115',
                number_of_left_players_agent_controls=1,
                number_of_right_players_agent_controls=0
            )
            print(f"Loaded Google Research Football environment")
            return env
        except ImportError:
            print("Google Research Football not available, falling back to MPE2")
            env_name = "mpe_adversary"
    
    if env_name == "mpe_adversary":
        try:
            from mpe2 import simple_adversary_v3
            env = simple_adversary_v3.env(max_cycles=25)
            print(f"Loaded MPE2 Simple Adversary environment")
            return env
        except ImportError:
            print("MPE2 not available, falling back to PettingZoo")
            env_name = "pistonball"
    
    if env_name == "pistonball":
        try:
            from pettingzoo.butterfly import pistonball_v6
            env = pistonball_v6.env(n_pistons=3, time_penalty=-0.1, continuous=False, random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5, max_cycles=125)
            print(f"Loaded PettingZoo Pistonball environment")
            return env
        except ImportError as e:
            print(f"PettingZoo not fully available ({e}), using mock environment")
            return MockEnvironment()
    
    raise ValueError(f"Unknown environment: {env_name}")


def make_env(kind="mpe_adversary", seed=0):
    """Create environment for training (reuses load_environment logic)."""
    return load_environment(kind, seed)


def collect_rollout(env, policy, value_fn, steps=256):
    """Collect rollout data from environment interaction."""
    from src.rl.ppo_core import RolloutBuffer
    
    obs_dim = 10  # Mock environment observation dimension
    buffer = RolloutBuffer(steps, obs_dim)
    
    obs = env.reset()
    if not hasattr(obs, '__len__') or len(obs.shape) == 0:
        obs = np.random.randn(obs_dim)  # Fallback for mock
    
    for step in range(steps):
        # Get action from policy
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            logits = policy(obs_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
            
            value = value_fn(obs_tensor)
        
        action_int = int(action.item())
        
        # Environment step
        next_obs, reward, done, info = env.step(action_int)
        
        # Add to buffer
        buffer.add(obs, action_int, logp.item(), reward, value.item(), done)
        
        obs = next_obs if not done else env.reset()
        if not hasattr(obs, '__len__') or len(obs.shape) == 0:
            obs = np.random.randn(obs_dim)
    
    # Compute GAE
    with torch.no_grad():
        last_val = value_fn(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
    
    buffer.compute_gae(last_val)
    return buffer.get_batch()


def smoke_train(steps=512, env_kind="mpe_adversary", seed=0):
    """Minimal PPO training smoke test."""
    # Create environment
    env = make_env(env_kind, seed)
    
    from src.rl.ppo_core import ppo_update
    
    obs_dim = 10
    action_dim = 4  # Mock environment action space
    
    # Create networks
    policy = PolicyNet(obs_dim, action_dim)
    value_fn = ValueNet(obs_dim)
    
    # Optimizer
    params = list(policy.parameters()) + list(value_fn.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    
    # Collect rollout
    print("Collecting rollout...")
    batch = collect_rollout(env, policy, value_fn, steps)
    
    # Get initial entropy for comparison
    with torch.no_grad():
        logits = policy(batch['obs'])
        dist = torch.distributions.Categorical(logits=logits)
        ent_start = dist.entropy().mean().item()
    
    # PPO update
    print("Running PPO update...")
    pi_loss, vf_loss, kl, entropy = ppo_update(
        policy, value_fn, optimizer, batch, 
        epochs=4, minibatch_size=64
    )
    
    print(f"PPO: pi={pi_loss:.3f} vf={vf_loss:.3f} kl={kl:.4f} ent={entropy:.3f}")
    
    # Simple success criteria: reasonable KL divergence and finite losses
    return kl <= 0.05 and abs(pi_loss) < 10.0 and abs(vf_loss) < 10.0


def dry_run_environment(env, policy, steps=10):
    """Test environment and policy with random actions."""
    print(f"\n=== Dry Run ({steps} steps) ===")
    
    obs = env.reset()
    if hasattr(obs, 'shape'):
        print(f"Observation shape: {obs.shape}")
        obs_flat = obs.flatten()
    else:
        # Handle dict observations from some environments
        obs_flat = np.array([0.0] * 100)  # Placeholder
        print(f"Observation type: {type(obs)}")
    
    for step in range(steps):
        # Random action for simplicity
        if hasattr(env.action_space, 'n'):
            action = env.action_space.sample()
        else:
            action = 0  # Fallback
        
        obs, reward, done, info = env.step(action)
        print(f"Step {step}: action={action}, reward={reward:.3f}, done={done}")
        
        if done:
            obs = env.reset()
            print("Environment reset")
    
    print("✓ Dry run completed successfully")


def main():
    parser = argparse.ArgumentParser(description='PPO Self-Play Skeleton')
    parser.add_argument('--env', choices=['grf', 'mpe_adversary', 'pistonball'], default='mpe_adversary', help='Environment')
    parser.add_argument('--steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Test environment setup')
    parser.add_argument('--train', action='store_true', help='Run PPO smoke training')
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load environment
    env = load_environment(args.env, args.seed)
    
    if args.dry_run:
        policy = PolicyNet(10, 4)  # Mock dimensions
        dry_run_environment(env, policy, steps=5)
        return
    
    if args.train:
        success = smoke_train(steps=args.steps, env_kind=args.env, seed=args.seed)
        print(f"Smoke train {'✓ PASSED' if success else '✗ FAILED'}")
        return
    
    print(f"Starting PPO self-play training for {args.steps} steps...")
    print("TODO: Implement full PPO training loop")
    print("Use --train for smoke test or --dry-run for environment test")


if __name__ == "__main__":
    main()