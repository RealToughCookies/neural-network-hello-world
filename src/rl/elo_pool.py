"""
v1 Elo-based Opponent Pool System

JSON schema (v1-elo-pool):
{
  "version": "v1-elo-pool", 
  "created_at": "2025-08-26T00:00:00Z",
  "config": { "elo_k": 32, "scale": 400, "initial": 1200 },
  "agents": [
    {
      "id": "ckpt-12000",
      "path": "artifacts/rl_ckpts_dota/ckpt_step_12000.pt",
      "kind": "v3|model_only",
      "roles": ["good","adv"],
      "elo": 1200.0,
      "games": 0, "wins": 0, "losses": 0, "draws": 0,
      "meta": {
        "env": "dota_last_hit",
        "global_step": 12000,
        "seed": 1337,
        "created_at": "2025-08-26T00:00:00Z"
      }
    }
  ]
}
"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


class OpponentPoolV1:
    """v1 Elo-based opponent pool with logistic expected scores."""
    
    def __init__(self, path: Path, config: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.config = config or {"elo_k": 32, "scale": 400, "initial": 1200}
        self.data = {
            "version": "v1-elo-pool",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": self.config,
            "agents": []
        }
        
    @staticmethod
    def is_supported(obj: Dict[str, Any]) -> bool:
        """Check if object is a v1-elo-pool."""
        return obj.get("version") == "v1-elo-pool"
    
    def load(self) -> None:
        """Load pool from JSON file."""
        if self.path.exists():
            with open(self.path, 'r') as f:
                self.data = json.load(f)
                # Ensure config exists
                if "config" not in self.data:
                    self.data["config"] = self.config
        else:
            # Create new pool
            self.data = {
                "version": "v1-elo-pool",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": self.config,
                "agents": []
            }
    
    def save(self) -> None:
        """Save pool to JSON file with atomic write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        import os
        os.replace(tmp_path, self.path)
    
    def add_or_update_agent(self, *, ckpt_path: str, ckpt_kind: str,
                           roles: List[str], meta: Dict[str, Any],
                           id_hint: Optional[str] = None) -> str:
        """Add or update agent in pool."""
        # Generate ID from hint or path
        if id_hint:
            agent_id = id_hint
        else:
            # Extract meaningful ID from path
            path_obj = Path(ckpt_path)
            if "step" in path_obj.name:
                agent_id = path_obj.stem
            else:
                agent_id = f"ckpt-{hash(ckpt_path) % 100000:05d}"
        
        # Find existing agent or create new
        agent = None
        for a in self.data["agents"]:
            if a["id"] == agent_id:
                agent = a
                break
        
        if agent is None:
            # New agent
            agent = {
                "id": agent_id,
                "path": ckpt_path,
                "kind": ckpt_kind,
                "roles": roles,
                "elo": float(self.data["config"]["initial"]),
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "meta": meta
            }
            self.data["agents"].append(agent)
        else:
            # Update existing
            agent["path"] = ckpt_path
            agent["kind"] = ckpt_kind
            agent["roles"] = roles
            agent["meta"] = meta
            
        return agent_id
    
    def expected(self, r_a: float, r_b: float) -> float:
        """Logistic expected score: P(A wins) = 1/(1+10^((R_b-R_a)/scale))"""
        scale = self.data["config"]["scale"]
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / scale))
    
    def record_result(self, *, learner_id: str, opp_id: str, result: float, 
                     k: Optional[int] = None) -> None:
        """
        Record game result and update Elo ratings.
        result: 1.0 = learner wins, 0.5 = draw, 0.0 = opponent wins
        """
        if k is None:
            k = self.data["config"]["elo_k"]
            
        # Find agents
        learner = None
        opponent = None
        for agent in self.data["agents"]:
            if agent["id"] == learner_id:
                learner = agent
            elif agent["id"] == opp_id:
                opponent = agent
                
        if learner is None or opponent is None:
            return  # Agent not found
            
        # Current ratings
        r_learner = learner["elo"]
        r_opponent = opponent["elo"]
        
        # Expected scores
        e_learner = self.expected(r_learner, r_opponent)
        e_opponent = self.expected(r_opponent, r_learner)
        
        # Update ratings
        learner["elo"] += k * (result - e_learner)
        opponent["elo"] += k * ((1.0 - result) - e_opponent)
        
        # Update game counts
        learner["games"] += 1
        opponent["games"] += 1
        
        if result == 1.0:
            learner["wins"] += 1
            opponent["losses"] += 1
        elif result == 0.0:
            learner["losses"] += 1
            opponent["wins"] += 1
        else:  # draw
            learner["draws"] += 1
            opponent["draws"] += 1
    
    def sample(self, *, strategy: str, topk: int = 5, tau: float = 1.5) -> Dict[str, Any]:
        """
        Sample opponent using specified strategy.
        
        Strategies:
        - 'uniform': random choice
        - 'topk': top-k by ELO, uniform within K  
        - 'pfsp_elo': PFSP-esque using Elo expected scores
        """
        if not self.data["agents"]:
            raise ValueError("No agents in pool")
            
        agents = self.data["agents"]
        
        if strategy == "uniform":
            return random.choice(agents)
            
        elif strategy == "topk":
            # Sort by Elo descending, take top-k
            sorted_agents = sorted(agents, key=lambda a: a["elo"], reverse=True)
            top_agents = sorted_agents[:min(topk, len(sorted_agents))]
            return random.choice(top_agents)
            
        elif strategy == "pfsp_elo":
            # PFSP-esque: p_i ∝ (1 - |0.5 - p_win_i|)^τ
            # Use Elo expected score vs average opponent as proxy
            avg_elo = sum(a["elo"] for a in agents) / len(agents)
            
            weights = []
            for agent in agents:
                p_win = self.expected(avg_elo, agent["elo"])  # Expected vs this agent
                difficulty = 1.0 - abs(0.5 - p_win)  # Closer to 0.5 = more interesting
                weight = difficulty ** tau
                weights.append(weight)
            
            # Weighted random choice
            total_weight = sum(weights)
            if total_weight == 0:
                return random.choice(agents)
                
            r = random.random() * total_weight
            cumsum = 0
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    return agents[i]
            return agents[-1]  # fallback
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    @staticmethod
    def migrate_legacy(legacy_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy pool format to v1-elo-pool."""
        config = {"elo_k": 32, "scale": 400, "initial": 1200}
        
        v1_agents = []
        for agent_data in legacy_obj.get("agents", []):
            v1_agent = {
                "id": agent_data.get("id", f"legacy-{len(v1_agents)}"),
                "path": agent_data.get("path", agent_data.get("ckpt_path", "")),
                "kind": "model_only",  # Assume legacy is model-only
                "roles": ["good", "adv"],  # Default roles
                "elo": float(agent_data.get("elo", config["initial"])),
                "games": agent_data.get("games", 0),
                "wins": agent_data.get("wins", 0),
                "losses": agent_data.get("losses", 0), 
                "draws": agent_data.get("draws", 0),
                "meta": {
                    "env": agent_data.get("env", "unknown"),
                    "global_step": agent_data.get("step", 0),
                    "seed": agent_data.get("seed", 0),
                    "created_at": agent_data.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                }
            }
            v1_agents.append(v1_agent)
            
        return {
            "version": "v1-elo-pool",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": config,
            "agents": v1_agents
        }