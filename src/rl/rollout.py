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
    counts = {r: 0 for r in set(roles.values())}
    # role-indexed traj buffers
    bufs: Dict[str, TrajBuf] = {r: TrajBuf([],[],[],[],[],[]) for r in counts.keys()}

    plan_idx = 0
    episode_idx = 0
    rng = np.random.default_rng(seed)

    def need_more() -> bool:
        return any(counts[r] < per_agent_steps for r in counts)

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

        # per-episode ep buffers for GAE
        ep_obs   = {ag: [] for ag in agents}
        ep_act   = {ag: [] for ag in agents}
        ep_logp  = {ag: [] for ag in agents}
        ep_val   = {ag: [] for ag in agents}
        ep_rew   = {ag: [] for ag in agents}
        ep_done  = {ag: [] for ag in agents}

        while not all(ts.dones.values()):
            actions = {}
            for ag in agents:
                role = roles[ag]
                o = ts.obs[ag]
                if role in obs_tf:
                    o = obs_tf[role](o)
                ep_obs[ag].append(o.copy())
                o_t = torch.from_numpy(o.astype(np.float32)).unsqueeze(0)
                logits = policy.act(role, o_t)
                probs = torch.softmax(logits, dim=-1)
                # sample (stochastic policy during rollout)
                dist = torch.distributions.Categorical(probs=probs)
                a = int(dist.sample().item())
                lp = float(dist.log_prob(torch.tensor(a)).item())
                v = float(policy.value(role, o_t).item())
                actions[ag] = a
                ep_act[ag].append(a); ep_logp[ag].append(lp); ep_val[ag].append(v)
            ts = adapter.step(actions)
            for ag in agents:
                ep_rew[ag].append(float(ts.rewards[ag]))
                ep_done[ag].append(bool(ts.dones[ag]))

        # push completed episode into role trajs
        for ag in agents:
            role = roles[ag]
            # only add until we reach the per-agent budget (truncate tail if needed)
            rem = per_agent_steps - counts[role]
            if rem <= 0: 
                continue
            take = min(rem, len(ep_act[ag]))
            sl = slice(0, take)
            bufs[role].obs.extend(ep_obs[ag][sl])
            bufs[role].act.extend(ep_act[ag][sl])
            bufs[role].logp.extend(ep_logp[ag][sl])
            bufs[role].val.extend(ep_val[ag][sl])
            bufs[role].rew.extend(ep_rew[ag][sl])
            bufs[role].done.extend(ep_done[ag][sl])
            counts[role] += take

    # compute per-role GAE/returns
    adv_out: Dict[str, torch.Tensor] = {}
    ret_out: Dict[str, torch.Tensor] = {}
    obs_out: Dict[str, torch.Tensor] = {}
    act_out: Dict[str, torch.Tensor] = {}
    logp_out: Dict[str, torch.Tensor] = {}
    val_out: Dict[str, torch.Tensor] = {}

    for role, buf in bufs.items():
        # convert to tensors
        obs_t  = _to_t(np.stack(buf.obs, axis=0))               # [T, obs_dim]
        act_t  = torch.tensor(buf.act, dtype=torch.long)        # [T]
        logp_t = _to_t(buf.logp)                                # [T]
        val_t  = _to_t(buf.val)                                 # [T]
        rew_t  = _to_t(buf.rew)                                 # [T]
        done_t = torch.tensor(buf.done, dtype=torch.float32)    # [T]

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

    return Batch(obs=obs_out, act=act_out, logp=logp_out, val=val_out, adv=adv_out, ret=ret_out), counts