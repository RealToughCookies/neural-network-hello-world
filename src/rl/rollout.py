from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Tuple, Any
import numpy as np
import torch

@dataclass
class TrajBuf:
    obs:   List[np.ndarray]
    act:   List[int]
    logp:  List[float]
    val:   List[float]
    rew:   List[float]
    done:  List[bool]

@dataclass
class Batch:
    # role -> tensors
    obs: Dict[str, torch.Tensor]
    act: Dict[str, torch.Tensor]
    logp: Dict[str, torch.Tensor]
    val: Dict[str, torch.Tensor]
    adv: Dict[str, torch.Tensor]
    ret: Dict[str, torch.Tensor]

def _to_t(x: np.ndarray | List[float], dtype=torch.float32):
    if isinstance(x, list): x = np.array(x, dtype=np.float32)
    if isinstance(x, np.ndarray) and x.dtype != np.float32:
        x = x.astype(np.float32)
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=dtype)

@torch.no_grad()
def collect_rollouts(
    *,
    adapter,                      # EnvAdapter
    policy,                       # MultiHeadPolicy with .act(role, x) and .value(role, x)
    roles: Dict[str, str],        # agent_name -> role ("good"/"adv")
    agents: List[str],
    per_agent_steps: int,
    seed: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    obs_transform: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
    match_plan: Optional[List[Tuple[str, Any]]] = None,  # list of ("self"|"pool", entry)
    load_opponent: Optional[Callable[[Any], None]] = None, # invoked when kind=="pool"
) -> Tuple[Batch, Dict[str,int]]:
    """
    Collect at least per_agent_steps transitions per role. Returns Batch and counts per role.
    match_plan: if provided, we iterate it and repeat if needed to fill budget.
    load_opponent(entry): optional callback to swap opponent head before an episode.
    """
    obs_tf = obs_transform or {}
    
    # Role-based aggregation setup
    target = int(per_agent_steps)
    roles_map = roles  # Dict[agent_id, role_str]
    role_keys = sorted(set(roles_map.values()))
    counts = {role: 0 for role in role_keys}
    buffers = {role: [] for role in role_keys}

    def role_of(agent_id):
        return roles_map[agent_id]

    plan_idx = 0
    episode_idx = 0
    rng = np.random.default_rng(seed)

    def need_more() -> bool:
        return min(counts.values()) < target

    while need_more():
        # select match
        if match_plan and len(match_plan) > 0:
            kind, entry = match_plan[plan_idx % len(match_plan)]
            plan_idx += 1
        else:
            kind, entry = ("self", None)
        if kind == "pool" and load_opponent is not None:
            load_opponent(entry)

        ts = adapter.reset(seed=seed + episode_idx)
        episode_idx += 1

        # Step-by-step collection with role-based aggregation
        while not all(ts.dones.values()) and need_more():
            # Generate actions for all active agents
            actions = {}
            for a in ts.obs.keys():  # these are AGENT NAMES, not roles
                if not ts.dones.get(a, True):  # Agent is alive
                    role = role_of(a)
                    o = ts.obs[a]
                    if role in obs_tf:
                        o = obs_tf[role](o)
                    
                    o_t = torch.from_numpy(o.astype(np.float32)).unsqueeze(0)
                    logits = policy.act(role, o_t)
                    probs = torch.softmax(logits, dim=-1)
                    # sample (stochastic policy during rollout)
                    dist = torch.distributions.Categorical(probs=probs)
                    actions[a] = int(dist.sample().item())
            
            # Step environment
            next_ts = adapter.step(actions)
            
            # Collect transitions per live agent, aggregate by role
            for a in ts.obs.keys():  # these are AGENT NAMES, not roles
                if not ts.dones.get(a, True):  # Agent was alive this tick
                    r = role_of(a)
                    # only store if this role still needs budget
                    if counts[r] < target:
                        # Build transition for this agent
                        o = ts.obs[a]
                        if r in obs_tf:
                            o = obs_tf[r](o)
                        
                        o_t = torch.from_numpy(o.astype(np.float32)).unsqueeze(0)
                        logits = policy.act(r, o_t)
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs=probs)
                        action = actions[a]
                        lp = float(dist.log_prob(torch.tensor(action)).item())
                        v = float(policy.value(r, o_t).item())
                        
                        # Store in role buffer
                        buffers[r].append({
                            'obs': o.copy(),
                            'act': action,
                            'logp': lp,
                            'val': v,
                            'rew': float(next_ts.rewards.get(a, 0.0)),
                            'done': bool(next_ts.dones.get(a, False))
                        })
                        counts[r] += 1
            
            ts = next_ts
            
            # Reset if episode ends but we still need more steps
            if all(ts.dones.values()) and need_more():
                ts = adapter.reset(seed=seed + episode_idx)
                episode_idx += 1

    # Safety: exact size per role
    for r in role_keys:
        if len(buffers[r]) > target:
            buffers[r] = buffers[r][:target]

    # compute per-role GAE/returns
    adv_out: Dict[str, torch.Tensor] = {}
    ret_out: Dict[str, torch.Tensor] = {}
    obs_out: Dict[str, torch.Tensor] = {}
    act_out: Dict[str, torch.Tensor] = {}
    logp_out: Dict[str, torch.Tensor] = {}
    val_out: Dict[str, torch.Tensor] = {}

    for role in role_keys:
        buf_list = buffers[role]
        if not buf_list:
            continue
            
        # Extract data from buffer list
        obs_list = [t['obs'] for t in buf_list]
        act_list = [t['act'] for t in buf_list]
        logp_list = [t['logp'] for t in buf_list]
        val_list = [t['val'] for t in buf_list]
        rew_list = [t['rew'] for t in buf_list]
        done_list = [t['done'] for t in buf_list]
        
        # convert to tensors
        obs_t  = _to_t(np.stack(obs_list, axis=0))               # [T, obs_dim]
        act_t  = torch.tensor(act_list, dtype=torch.long)        # [T]
        logp_t = _to_t(logp_list)                                # [T]
        val_t  = _to_t(val_list)                                 # [T]
        rew_t  = _to_t(rew_list)                                 # [T]
        done_t = torch.tensor(done_list, dtype=torch.float32)    # [T]

        # GAE (no bootstrap across episodes; done=1 terminates)
        T = rew_t.shape[0]
        adv = torch.zeros(T, dtype=torch.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(done_t[t].item())
            next_val = float(val_t[t+1].item()) if t+1 < T and nonterminal > 0.5 else 0.0
            delta = float(rew_t[t].item()) + gamma * next_val * nonterminal - float(val_t[t].item())
            lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
        ret = adv + val_t

        adv_out[role] = adv
        ret_out[role] = ret
        obs_out[role] = obs_t
        act_out[role] = act_t
        logp_out[role] = logp_t
        val_out[role] = val_t

    # Log and assert exact budget fulfillment
    import logging
    logger = logging.getLogger(__name__)
    msg = ", ".join(f"{r}={counts[r]}" for r in role_keys)
    logger.info("[collector] adapter_native | per-role steps: %s", msg)
    assert all(counts[r] == target for r in role_keys), f"collector budget miss: {counts} vs target={target}"
    
    return Batch(obs=obs_out, act=act_out, logp=logp_out, val=val_out, adv=adv_out, ret=ret_out), counts