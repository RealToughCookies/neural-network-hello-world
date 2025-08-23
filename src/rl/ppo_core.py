"""
PPO core implementation with GAE and clipped policy loss.
Based on Schulman et al. 2017 (PPO) and Schulman et al. 2015 (GAE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RolloutBuffer:
    """Buffer for collecting rollout data and computing GAE."""
    
    def __init__(self, size, obs_dim, device="cpu"):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        
        # Rollout storage
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.acts = torch.zeros((size,), dtype=torch.long, device=device)
        self.logps = torch.zeros((size,), dtype=torch.float32, device=device)
        self.rews = torch.zeros((size,), dtype=torch.float32, device=device)
        self.vals = torch.zeros((size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        
        # GAE outputs
        self.advs = torch.zeros((size,), dtype=torch.float32, device=device)
        self.rets = torch.zeros((size,), dtype=torch.float32, device=device)
    
    def add(self, obs, act, logp, rew, val, done):
        """Add a transition to the buffer."""
        self.obs[self.ptr] = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.acts[self.ptr] = torch.tensor(act, dtype=torch.long, device=self.device)
        self.logps[self.ptr] = torch.tensor(logp, dtype=torch.float32, device=self.device)
        self.rews[self.ptr] = torch.tensor(rew, dtype=torch.float32, device=self.device)
        self.vals[self.ptr] = torch.tensor(val, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    
    def compute_gae(self, last_val=0.0, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        size = self.size if self.full else self.ptr
        
        # Bootstrap values: V(s_{t+1}) = V_next if not done, else 0
        vals_next = torch.cat([self.vals[1:size], torch.tensor([last_val], device=self.device)])
        vals_next = vals_next * (1 - self.dones[:size])  # Zero out if done
        
        # TD errors: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        deltas = self.rews[:size] + gamma * vals_next - self.vals[:size]
        
        # GAE: A_t = δ_t + γλ A_{t+1}
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(size)):
            gae = deltas[t] + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
        
        # Returns: R_t = A_t + V(s_t)
        returns = advantages + self.vals[:size]
        
        self.advs[:size] = advantages
        self.rets[:size] = returns
        
        return advantages, returns
    
    def get_batch(self):
        """Get current batch data."""
        size = self.size if self.full else self.ptr
        return {
            'obs': self.obs[:size],
            'acts': self.acts[:size],
            'logps': self.logps[:size],
            'advs': self.advs[:size],
            'rets': self.rets[:size],
            'vals': self.vals[:size]
        }


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Standalone GAE computation for arrays."""
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.tensor(values, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    
    # Bootstrap next values
    vals_next = torch.cat([values[1:], torch.tensor([0.0])])
    vals_next = vals_next * (1 - dones)
    
    # TD errors
    deltas = rewards + gamma * vals_next - values
    
    # GAE computation
    advantages = torch.zeros_like(deltas)
    gae = 0.0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, value_fn, optimizer, batch, clip=0.2, vf_coef=0.5, ent_coef=0.01, 
               epochs=4, minibatch_size=64, target_kl=0.03):
    """PPO update with clipped policy loss and value function loss."""
    
    obs, acts, logps_old, advs, rets, vals_old = (
        batch['obs'], batch['acts'], batch['logps'], 
        batch['advs'], batch['rets'], batch['vals']
    )
    
    # Standardize advantages
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    
    batch_size = len(obs)
    indices = torch.randperm(batch_size)
    
    for epoch in range(epochs):
        epoch_pi_loss, epoch_vf_loss, epoch_kl, epoch_ent = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        
        # Minibatch updates
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            mb_obs = obs[mb_indices]
            mb_acts = acts[mb_indices]
            mb_logps_old = logps_old[mb_indices]
            mb_advs = advs[mb_indices]
            mb_rets = rets[mb_indices]
            
            # Forward pass
            logits = policy(mb_obs)
            dist = torch.distributions.Categorical(logits=logits)
            logps_new = dist.log_prob(mb_acts)
            entropy = dist.entropy()
            
            values_new = value_fn(mb_obs).squeeze()
            
            # PPO policy loss
            ratio = torch.exp(logps_new - mb_logps_old)
            surr1 = ratio * mb_advs
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_advs
            pi_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            vf_loss = F.mse_loss(values_new, mb_rets) * vf_coef
            
            # Entropy bonus
            ent_loss = -entropy.mean() * ent_coef
            
            # Total loss
            loss = pi_loss + vf_loss + ent_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_fn.parameters()), 0.5
            )
            optimizer.step()
            
            # Logging
            with torch.no_grad():
                approx_kl = (mb_logps_old - logps_new).mean().item()
                epoch_pi_loss += pi_loss.item()
                epoch_vf_loss += vf_loss.item()
                epoch_kl += approx_kl
                epoch_ent += entropy.mean().item()
                n_batches += 1
        
        # Early stopping if KL divergence too high
        avg_kl = epoch_kl / n_batches
        if avg_kl > target_kl * 1.5:
            break
    
    return (
        epoch_pi_loss / n_batches,
        epoch_vf_loss / n_batches, 
        avg_kl,
        epoch_ent / n_batches
    )