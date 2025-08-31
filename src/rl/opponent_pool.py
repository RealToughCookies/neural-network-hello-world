"""
Opponent pool management with Elo ratings and sampling strategies.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .elo import expected_score, k_factor, update


def load_pool(path: Path) -> dict:
    """
    Load opponent pool from JSON file. Creates empty pool if missing.
    Migrates schema 1 to schema 2 automatically.
    
    Args:
        path: Path to pool JSON file
        
    Returns:
        Pool dictionary with schema version 2
    """
    if not path.exists():
        pool = {
            "schema": 2,
            "env": "",
            "roles": [],
            "agents": [],
            "matches": []
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        save_pool(path, pool)
        return pool
    
    with open(path, 'r') as f:
        pool = json.load(f)
    
    # Migrate schema 1 to schema 2
    if pool.get("schema") == 1:
        pool["schema"] = 2
        pool["matches"] = []
        save_pool(path, pool)  # Auto-save migration
    
    return pool


def save_pool(path: Path, pool: dict) -> None:
    """
    Save pool to JSON file with atomic write.
    
    Args:
        path: Path to pool JSON file
        pool: Pool dictionary to save
    """
    tmp_path = path.with_suffix('.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(pool, f, indent=2)
    os.replace(tmp_path, path)


def register_snapshot(pool: dict, env: str, role: str, ckpt_path: Path, default_elo: float = 1000.0) -> None:
    """
    Register a checkpoint snapshot in the pool. No-op if policy_id already exists.
    
    Args:
        pool: Pool dictionary to modify
        env: Environment name
        role: Role name (e.g., "good", "adv")
        ckpt_path: Path to checkpoint file
        default_elo: Initial Elo rating for new agents
    """
    policy_id = f"{ckpt_path.stem}:{role}"
    
    # Check if already registered
    for agent in pool["agents"]:
        if agent["policy_id"] == policy_id:
            return
    
    # Update pool metadata
    if not pool["env"]:
        pool["env"] = env
    if role not in pool["roles"]:
        pool["roles"].append(role)
        pool["roles"].sort()
    
    # Add new agent
    agent = {
        "policy_id": policy_id,
        "role": role,
        "ckpt": str(ckpt_path),
        "elo": default_elo,
        "games": 0,
        "wins": 0,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    pool["agents"].append(agent)


def by_role(pool: dict, role: str) -> List[dict]:
    """
    Get all agents for a specific role.
    
    Args:
        pool: Pool dictionary
        role: Role name
        
    Returns:
        List of agent dictionaries for the role
    """
    return [agent for agent in pool["agents"] if agent["role"] == role]


def sample_uniform(pool: dict, role: str, n: int, rng: np.random.Generator) -> List[dict]:
    """
    Sample n agents uniformly from role.
    
    Args:
        pool: Pool dictionary
        role: Role to sample from
        n: Number of samples (with replacement)
        rng: Random number generator
        
    Returns:
        List of sampled agent dictionaries
    """
    agents = by_role(pool, role)
    if not agents:
        return []
    
    indices = rng.integers(0, len(agents), size=n)
    return [agents[i] for i in indices]


def sample_topk(pool: dict, role: str, k: int, n: int, rng: np.random.Generator) -> List[dict]:
    """
    Sample from top-k agents by Elo rating.
    
    Args:
        pool: Pool dictionary
        role: Role to sample from
        k: Number of top agents to consider
        n: Number of samples (with replacement)
        rng: Random number generator
        
    Returns:
        List of sampled agent dictionaries
    """
    agents = by_role(pool, role)
    if not agents:
        return []
    
    # Sort by Elo descending and take top k
    agents_sorted = sorted(agents, key=lambda x: x["elo"], reverse=True)
    top_k = agents_sorted[:min(k, len(agents_sorted))]
    
    if not top_k:
        return []
    
    indices = rng.integers(0, len(top_k), size=n)
    return [top_k[i] for i in indices]


def sample_pfsp_elo(pool: dict, role: str, agent_elo: float, mode: str = "even", p: float = 2.0, n: int = 1, rng: np.random.Generator = None) -> List[dict]:
    """
    Sample using PFSP-Elo weighting based on expected win probability.
    
    Args:
        pool: Pool dictionary
        role: Role to sample from
        agent_elo: Elo rating of the current agent
        mode: Weighting mode ("hard", "even", "easy")
        p: Power parameter for weighting function
        n: Number of samples (with replacement)
        rng: Random number generator
        
    Returns:
        List of sampled agent dictionaries
    """
    agents = by_role(pool, role)
    if not agents:
        return []
    
    # Compute weights based on expected scores
    weights = []
    for agent in agents:
        x = expected_score(agent_elo, agent["elo"])
        
        if mode == "hard":
            w = (1.0 - x) ** p  # Prioritize hard opponents
        elif mode == "even":
            w = (0.5 - abs(x - 0.5)) ** p  # Peak near 0.5 expected score
        elif mode == "easy":
            w = x ** p  # Prioritize easy opponents
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        weights.append(w)
    
    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(agents)) / len(agents)
    
    # Sample with replacement
    if rng is None:
        rng = np.random.default_rng()
    
    indices = rng.choice(len(agents), size=n, p=weights, replace=True)
    return [agents[i] for i in indices]


def record_match(pool: dict, agent_a_id: str, agent_b_id: str, score_a: float, timestamp: str = None) -> None:
    """
    Record a match result and update Elo ratings for both agents.
    
    Args:
        pool: Pool dictionary to update
        agent_a_id: Policy ID of agent A
        agent_b_id: Policy ID of agent B  
        score_a: Score for agent A (1.0 = win, 0.5 = draw, 0.0 = loss)
        timestamp: ISO8601 timestamp (auto-generated if None)
    """
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    # Find agents
    agent_a = None
    agent_b = None
    for agent in pool["agents"]:
        if agent["policy_id"] == agent_a_id:
            agent_a = agent
        elif agent["policy_id"] == agent_b_id:
            agent_b = agent
    
    if agent_a is None or agent_b is None:
        raise ValueError(f"Agent not found: {agent_a_id if agent_a is None else agent_b_id}")
    
    # Get current ratings
    elo_a = agent_a["elo"] 
    elo_b = agent_b["elo"]
    
    # Compute K-factors
    k_a = k_factor(elo_a, agent_a["games"])
    k_b = k_factor(elo_b, agent_b["games"])
    
    # Update ratings (use same K-factor for consistency)
    k_avg = (k_a + k_b) / 2.0
    new_elo_a, new_elo_b = update(elo_a, elo_b, score_a, k_avg)
    
    # Update agent records
    agent_a["elo"] = new_elo_a
    agent_a["games"] += 1
    if score_a == 1.0:
        agent_a["wins"] += 1
    
    agent_b["elo"] = new_elo_b  
    agent_b["games"] += 1
    if score_a == 0.0:
        agent_b["wins"] += 1
    
    # Record match
    match_record = {
        "timestamp": timestamp,
        "agent_a": agent_a_id,
        "agent_b": agent_b_id,
        "score_a": score_a,
        "elo_a_before": elo_a,
        "elo_b_before": elo_b,
        "elo_a_after": new_elo_a,
        "elo_b_after": new_elo_b
    }
    pool["matches"].append(match_record)