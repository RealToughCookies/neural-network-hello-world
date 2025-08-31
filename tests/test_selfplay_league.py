import pytest
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rl.ppo_selfplay_skeleton import pick_opponent_ckpt


@pytest.mark.unit
def test_pick_opponent_ckpt_uniform():
    """Test uniform opponent sampling."""
    # Create fake pool
    pool = {
        "schema": 2,
        "env": "test_env", 
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "agent_1:good",
                "role": "good",
                "ckpt": "/path/to/agent1.pt",
                "elo": 1200.0,
                "games": 10,
                "wins": 6
            },
            {
                "policy_id": "agent_2:good", 
                "role": "good",
                "ckpt": "/path/to/agent2.pt",
                "elo": 1000.0,
                "games": 5,
                "wins": 2
            },
            {
                "policy_id": "agent_3:adv",
                "role": "adv",
                "ckpt": "/path/to/agent3.pt", 
                "elo": 1100.0,
                "games": 8,
                "wins": 4
            }
        ],
        "matches": []
    }
    
    # Create seeded RNG for deterministic results
    rng = np.random.Generator(np.random.PCG64(42))
    
    # Test uniform sampling for good role
    ckpt_path, agent = pick_opponent_ckpt(
        pool, "good", "uniform",
        topk=5, mode="even", p=2.0, agent_elo=1000.0, rng=rng
    )
    
    assert ckpt_path is not None
    assert agent is not None
    assert agent["role"] == "good"
    assert ckpt_path == agent["ckpt"]


@pytest.mark.unit
def test_pick_opponent_ckpt_topk():
    """Test top-k opponent sampling."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "weak:good",
                "role": "good", 
                "ckpt": "/weak.pt",
                "elo": 900.0,
                "games": 10,
                "wins": 3
            },
            {
                "policy_id": "strong:good",
                "role": "good",
                "ckpt": "/strong.pt", 
                "elo": 1400.0,
                "games": 15,
                "wins": 12
            },
            {
                "policy_id": "medium:good",
                "role": "good",
                "ckpt": "/medium.pt",
                "elo": 1100.0, 
                "games": 8,
                "wins": 5
            }
        ],
        "matches": []
    }
    
    rng = np.random.Generator(np.random.PCG64(123))
    
    # Test top-2 sampling - should only get strong or medium agents
    selected_agents = []
    for _ in range(10):  # Sample multiple times
        ckpt_path, agent = pick_opponent_ckpt(
            pool, "good", "topk", 
            topk=2, mode="even", p=2.0, agent_elo=1000.0, rng=rng
        )
        if agent:
            selected_agents.append(agent["policy_id"])
    
    # Should never sample the weak agent (lowest Elo)
    assert "weak:good" not in selected_agents
    # Should sample from top 2 agents
    assert any(agent_id in ["strong:good", "medium:good"] for agent_id in selected_agents)


@pytest.mark.unit
def test_pick_opponent_ckpt_pfsp():
    """Test PFSP opponent sampling."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good", "adv"],
        "agents": [
            {
                "policy_id": "easy:adv",
                "role": "adv",
                "ckpt": "/easy.pt",
                "elo": 800.0,  # Much weaker than agent
                "games": 5,
                "wins": 1
            },
            {
                "policy_id": "balanced:adv", 
                "role": "adv",
                "ckpt": "/balanced.pt",
                "elo": 1000.0,  # Equal to agent
                "games": 10,
                "wins": 5
            },
            {
                "policy_id": "hard:adv",
                "role": "adv", 
                "ckpt": "/hard.pt",
                "elo": 1200.0,  # Much stronger than agent
                "games": 15,
                "wins": 12
            }
        ],
        "matches": []
    }
    
    rng = np.random.Generator(np.random.PCG64(456))
    agent_elo = 1000.0
    
    # Test "hard" mode - should prefer strong opponents
    hard_selections = []
    for _ in range(50):
        ckpt_path, agent = pick_opponent_ckpt(
            pool, "adv", "pfsp",
            topk=5, mode="hard", p=2.0, agent_elo=agent_elo, rng=rng
        )
        if agent:
            hard_selections.append(agent["policy_id"])
    
    # Should prefer hard opponents in hard mode
    hard_count = hard_selections.count("hard:adv")
    easy_count = hard_selections.count("easy:adv")
    assert hard_count > easy_count, f"Hard mode should prefer strong opponents: hard={hard_count}, easy={easy_count}"
    
    # Test "easy" mode - should prefer weak opponents  
    easy_selections = []
    for _ in range(50):
        ckpt_path, agent = pick_opponent_ckpt(
            pool, "adv", "pfsp",
            topk=5, mode="easy", p=2.0, agent_elo=agent_elo, rng=rng
        )
        if agent:
            easy_selections.append(agent["policy_id"])
    
    easy_hard_count = easy_selections.count("hard:adv")
    easy_easy_count = easy_selections.count("easy:adv")
    assert easy_easy_count > easy_hard_count, f"Easy mode should prefer weak opponents: easy={easy_easy_count}, hard={easy_hard_count}"


@pytest.mark.unit  
def test_pick_opponent_ckpt_empty_pool():
    """Test behavior with empty pool."""
    empty_pool = {
        "schema": 2,
        "env": "test_env",
        "roles": [],
        "agents": [],
        "matches": []
    }
    
    rng = np.random.Generator(np.random.PCG64(789))
    
    ckpt_path, agent = pick_opponent_ckpt(
        empty_pool, "good", "uniform",
        topk=5, mode="even", p=2.0, agent_elo=1000.0, rng=rng
    )
    
    assert ckpt_path is None
    assert agent is None


@pytest.mark.unit
def test_pick_opponent_ckpt_no_role():
    """Test behavior when no agents of requested role exist."""
    pool = {
        "schema": 2,
        "env": "test_env",
        "roles": ["good"],
        "agents": [
            {
                "policy_id": "only_good:good",
                "role": "good",
                "ckpt": "/good.pt",
                "elo": 1000.0,
                "games": 10,
                "wins": 5
            }
        ],
        "matches": []
    }
    
    rng = np.random.Generator(np.random.PCG64(999))
    
    # Try to sample adversary when none exist
    ckpt_path, agent = pick_opponent_ckpt(
        pool, "adv", "uniform",
        topk=5, mode="even", p=2.0, agent_elo=1000.0, rng=rng
    )
    
    assert ckpt_path is None
    assert agent is None


@pytest.mark.unit
def test_pick_opponent_ckpt_invalid_strategy():
    """Test behavior with invalid strategy."""
    pool = {
        "schema": 2, 
        "env": "test_env",
        "roles": ["good"],
        "agents": [
            {
                "policy_id": "test:good",
                "role": "good",
                "ckpt": "/test.pt",
                "elo": 1000.0,
                "games": 10,
                "wins": 5
            }
        ],
        "matches": []
    }
    
    rng = np.random.Generator(np.random.PCG64(111))
    
    ckpt_path, agent = pick_opponent_ckpt(
        pool, "good", "invalid_strategy",
        topk=5, mode="even", p=2.0, agent_elo=1000.0, rng=rng
    )
    
    assert ckpt_path is None
    assert agent is None


@pytest.mark.unit
def test_opponent_sampling_distributions():
    """Test that sampling strategies produce expected distributions."""
    # Create pool with agents at different Elo levels
    pool = {
        "schema": 2,
        "env": "test_env", 
        "roles": ["good"],
        "agents": []
    }
    
    # Add agents with known Elo distribution
    elos = [800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    for i, elo in enumerate(elos):
        pool["agents"].append({
            "policy_id": f"agent_{i}:good",
            "role": "good",
            "ckpt": f"/agent_{i}.pt", 
            "elo": float(elo),
            "games": 10,
            "wins": 5
        })
    
    rng = np.random.Generator(np.random.PCG64(222))
    
    # Test uniform distribution
    uniform_selections = {}
    for _ in range(200):
        ckpt_path, agent = pick_opponent_ckpt(
            pool, "good", "uniform",
            topk=5, mode="even", p=2.0, agent_elo=1000.0, rng=rng
        )
        if agent:
            agent_id = agent["policy_id"]
            uniform_selections[agent_id] = uniform_selections.get(agent_id, 0) + 1
    
    # Uniform should select all agents roughly equally
    selection_counts = list(uniform_selections.values())
    if selection_counts:
        std_uniform = np.std(selection_counts)
        mean_uniform = np.mean(selection_counts)
        # Coefficient of variation should be low for uniform sampling
        cv_uniform = std_uniform / mean_uniform if mean_uniform > 0 else float('inf')
        assert cv_uniform < 0.5, f"Uniform sampling too uneven: CV={cv_uniform}"
    
    # Test top-k distribution (top 3)
    topk_selections = {}
    for _ in range(200):
        ckpt_path, agent = pick_opponent_ckpt(
            pool, "good", "topk",
            topk=3, mode="even", p=2.0, agent_elo=1000.0, rng=rng
        )
        if agent:
            agent_id = agent["policy_id"]
            topk_selections[agent_id] = topk_selections.get(agent_id, 0) + 1
    
    # Top-k should only select from highest Elo agents
    selected_elos = []
    for agent_id, count in topk_selections.items():
        if count > 0:
            agent_idx = int(agent_id.split('_')[1].split(':')[0])
            selected_elos.append(elos[agent_idx])
    
    if selected_elos:
        min_selected_elo = min(selected_elos)
        # Should only select from top 3 Elos: 1300, 1400, 1500
        assert min_selected_elo >= 1300, f"Top-3 selected agent with Elo {min_selected_elo} < 1300"