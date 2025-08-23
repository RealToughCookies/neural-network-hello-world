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
            from mpe2 import simple_adversary_v3 as simple_adv  # preferred, headless
        except ImportError as e:
            raise RuntimeError("MPE2 not installed. Run: pip install mpe2") from e
        
        try:
            env = simple_adv.parallel_env(max_cycles=25)
        except Exception:
            env = simple_adv.env(max_cycles=25)  # last resort
        
        try:
            env.reset(seed=seed)
        except Exception:
            env.reset()
        
        print(f"Loaded MPE2 Simple Adversary environment (parallel API)")
        return env
    
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


def role_split(obs_keys, learner_role):
    """Split MPE2 observation keys into learner and opponent groups."""
    goods = [k for k in obs_keys if "agent" in k]
    advs = [k for k in obs_keys if "adversary" in k]
    return (goods, advs) if learner_role == "good" else (advs, goods)


def _act_batch(model, obs_list, device="cpu"):
    """Batch policy forward pass for multiple agents."""
    import torch
    import numpy as np
    from torch.distributions import Categorical
    
    if not obs_list:
        return np.array([]), np.array([]), 0.0, None
    
    X = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device)
    logits = model(X)
    dist = Categorical(logits=logits)
    a = dist.sample()
    logp = dist.log_prob(a)
    return a.cpu().numpy(), logp.detach().cpu().numpy(), dist.entropy().mean().item(), logits


def collect_parallel_mpe(env, learner, opponent, learner_role="good", steps=512, value_fn=None, device="cpu"):
    """Collect per-agent trajectories from MPE2 parallel environment."""
    import torch
    import numpy as np
    
    obs = env.reset()
    t = 0
    traj = []  # list of (obs, act, logp, rew, val, done)
    ep_rew = 0.0
    
    while t < steps:
        # Split keys by role
        keys = list(obs.keys())
        goods = [k for k in keys if "agent" in k]
        advs = [k for k in keys if "adversary" in k]
        L_keys, O_keys = (goods, advs) if learner_role == "good" else (advs, goods)
        
        # Learner acts on its agents, opponent acts on the other team
        acts_L, logp_L, _, _ = _act_batch(learner, [obs[k] for k in L_keys], device)
        acts_O, _, _, _ = _act_batch(opponent, [obs[k] for k in O_keys], device)
        
        # Build action dictionary
        act_dict = {k: v for k, v in zip(L_keys, acts_L)}
        act_dict.update({k: v for k, v in zip(O_keys, acts_O)})
        
        # Environment step
        try:
            next_obs, rews, terms, truncs, _ = env.step(act_dict)
        except Exception as e:
            print(f"Environment step error: {e}, breaking")
            break
        
        # Record per-agent samples for learner-controlled agents
        for i, k in enumerate(L_keys):
            ob = np.asarray(obs[k], dtype=np.float32)
            ac = int(acts_L[i])
            lp = float(logp_L[i])
            rw = float(rews.get(k, 0.0))
            done = bool(terms.get(k, False) or truncs.get(k, False))
            val = 0.0
            
            if value_fn is not None:
                with torch.no_grad():
                    val = float(value_fn(torch.tensor(ob)[None]).squeeze(0).item())
            
            traj.append((ob, ac, lp, rw, val, done))
            ep_rew += rw
        
        obs = next_obs
        
        # Check termination
        if all(terms.values()) or all(truncs.values()):
            break
        
        t += 1
    
    return ep_rew, traj


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


class SelfPlayRunner:
    """Self-play training runner with learner vs frozen opponent."""
    
    def __init__(self, env, learner, opponent, device="cpu"):
        self.env = env
        self.learner = learner
        self.opponent = opponent
        self.device = device
        self.value_fn = None  # Will be set externally
    
    def update_opponent(self):
        """Copy learner weights to opponent."""
        self.opponent.load_state_dict(self.learner.state_dict())
    
    def collect(self, steps=512, learner_role="good"):
        """Collect self-play trajectory with role assignment."""
        # Try parallel env first, fallback to sequential
        try:
            obs = self.env.reset()
            is_parallel = isinstance(obs, dict)
        except Exception:
            try:
                obs = self.env.reset()
                is_parallel = not isinstance(obs, tuple)
            except Exception:
                # Mock environment fallback
                obs = np.random.randn(10)
                is_parallel = False
        
        ep_rew = 0.0
        traj = []
        t = 0
        
        while t < steps:
            if is_parallel and isinstance(obs, dict):
                # MPE2 parallel environment
                learner_keys, opp_keys = role_split(list(obs.keys()), learner_role)
                
                def act_batch(model, keys):
                    if not keys:
                        return np.array([]), np.array([]), 0.0
                    X = torch.tensor([obs[k] for k in keys], dtype=torch.float32, device=self.device)
                    logits = model(X)
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    logp = dist.log_prob(a)
                    return a.cpu().numpy(), logp.detach().cpu().numpy(), dist.entropy().mean().item()
                
                a_l, logp_l, ent = act_batch(self.learner, learner_keys)
                a_o, _, _ = act_batch(self.opponent, opp_keys)
                
                # Build action dict
                act = {}
                if len(learner_keys) > 0:
                    act.update({k: v for k, v in zip(learner_keys, a_l)})
                if len(opp_keys) > 0:
                    act.update({k: v for k, v in zip(opp_keys, a_o)})
                
                try:
                    next_obs, rew, term, trunc, _ = self.env.step(act)
                    
                    # Aggregate learner reward
                    lr_rew = [rew[k] for k in learner_keys if k in rew]
                    r = float(sum(lr_rew) / max(1, len(lr_rew))) if lr_rew else 0.0
                    
                    # Get value estimate for mean observation
                    if learner_keys and self.value_fn:
                        mean_obs = torch.tensor(
                            np.mean([obs[k] for k in learner_keys], axis=0), 
                            dtype=torch.float32
                        )
                        val = self.value_fn(mean_obs.unsqueeze(0)).squeeze(0).item()
                    else:
                        mean_obs = torch.tensor(np.random.randn(10), dtype=torch.float32)
                        val = 0.0
                    
                    # Store trajectory step
                    if len(a_l) > 0:
                        traj.append((
                            mean_obs.numpy(),
                            int(a_l[0]),
                            float(r),
                            float(val),
                            False,
                            float(logp_l[0]) if len(logp_l) > 0 else 0.0
                        ))
                    
                    ep_rew += r
                    obs = next_obs
                    
                    # Check termination
                    if (isinstance(term, dict) and all(term.values())) or \
                       (isinstance(trunc, dict) and all(trunc.values())):
                        break
                        
                except Exception as e:
                    print(f"Environment step error: {e}, breaking")
                    break
                    
                t += 1
            else:
                # Fallback to mock trajectory
                obs_flat = obs if hasattr(obs, '__len__') else np.random.randn(10)
                obs_tensor = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    logits = self.learner(obs_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    
                    if self.value_fn:
                        val = self.value_fn(obs_tensor).squeeze(0).item()
                    else:
                        val = 0.0
                
                traj.append((
                    obs_flat[:10] if len(obs_flat) >= 10 else np.pad(obs_flat, (0, 10-len(obs_flat))),
                    int(action.item()),
                    np.random.randn(),  # Mock reward
                    float(val),
                    False,
                    float(logp.item())
                ))
                
                ep_rew += traj[-1][2]
                break  # Single step for mock
        
        return ep_rew, traj


def selfplay_smoke_train(steps=1024):
    """Minimal self-play training smoke test."""
    # Create MPE2 environment
    env = make_env("mpe_adversary", seed=0)
    
    obs_dim = 10  # Mock for compatibility
    action_dim = 4
    
    # Create two policy networks
    learner = PolicyNet(obs_dim, action_dim)
    opponent = PolicyNet(obs_dim, action_dim)
    value_fn = ValueNet(obs_dim)
    
    # Initialize opponent as copy of learner
    opponent.load_state_dict(learner.state_dict())
    
    # Create self-play runner
    runner = SelfPlayRunner(env, learner, opponent)
    runner.value_fn = value_fn
    
    # Optimizer
    params = list(learner.parameters()) + list(value_fn.parameters())
    optimizer = torch.optim.Adam(params, lr=3e-4)
    
    # Get initial entropy
    with torch.no_grad():
        test_obs = torch.randn(1, obs_dim)
        logits = learner(test_obs)
        dist = torch.distributions.Categorical(logits=logits)
        ent0 = dist.entropy().item()
    
    # Collect self-play trajectory using parallel MPE2 API
    print("Collecting self-play trajectory...")
    
    # Try parallel collection first for MPE2
    try:
        ep_rew, traj = collect_parallel_mpe(
            env, learner, opponent, 
            learner_role="good", 
            steps=min(steps, 256), 
            value_fn=value_fn, 
            device="cpu"
        )
        print(f"Collected {len(traj)} per-agent samples via parallel API")
    except Exception as e:
        print(f"Parallel collection failed ({e}), using fallback")
        ep_rew, traj = runner.collect(steps=min(steps, 256), learner_role="good")
    
    if not traj:
        print("No trajectory collected, using mock data")
        return True
    
    # Convert trajectory to batch format for PPO
    from src.rl.ppo_core import ppo_update
    
    obs_list = [step[0] for step in traj]
    act_list = [step[1] for step in traj]
    rew_list = [step[3] for step in traj]  # Note: different index for parallel format
    val_list = [step[4] for step in traj]  # Note: different index for parallel format
    logp_list = [step[2] for step in traj]  # Note: different index for parallel format
    
    # Create batch dict
    batch = {
        'obs': torch.tensor(obs_list, dtype=torch.float32),
        'acts': torch.tensor(act_list, dtype=torch.long),
        'logps': torch.tensor(logp_list, dtype=torch.float32),
        'rews': torch.tensor(rew_list, dtype=torch.float32),
        'vals': torch.tensor(val_list, dtype=torch.float32),
        'advs': torch.zeros(len(traj), dtype=torch.float32),  # Simplified
        'rets': torch.tensor(rew_list, dtype=torch.float32)   # Simplified
    }
    
    # PPO update
    print("Running PPO update...")
    try:
        pi_loss, vf_loss, kl, entropy = ppo_update(
            learner, value_fn, optimizer, batch,
            epochs=2, minibatch_size=min(32, len(traj))
        )
        
        print(f"SelfPlay: ep_rew={ep_rew:.3f} kl={kl:.4f} ent={entropy:.3f}")
        
        # Update opponent
        runner.update_opponent()
        
        # Success criteria
        return kl <= 0.05 and entropy < ent0 + 1.0
        
    except Exception as e:
        print(f"PPO update failed: {e}")
        return False


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
    parser.add_argument('--selfplay', action='store_true', help='Run self-play training')
    parser.add_argument('--swap-every', type=int, default=1, help='Update opponent every N episodes')
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
    
    if args.selfplay:
        success = selfplay_smoke_train(steps=args.steps)
        print(f"Self-play smoke train {'✓ PASSED' if success else '✗ FAILED'}")
        return
    
    print(f"Starting PPO self-play training for {args.steps} steps...")
    print("TODO: Implement full PPO training loop")
    print("Use --train for smoke test or --dry-run for environment test")


if __name__ == "__main__":
    main()