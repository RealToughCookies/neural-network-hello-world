from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from src.rl.env_api import Timestep, register

@register("dota_last_hit")
class DotaLastHitAdapter:
    def __init__(self,
                 hp0: float = 550.0,
                 hp_decay: float = 10.0,
                 base_cd: int = 6,           # ticks
                 max_steps: int = 200,
                 render_mode: str | None = None):
        self.hp0 = hp0
        self.hp_decay = hp_decay
        self.base_cd = base_cd
        self.max_steps = max_steps
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._hp = hp0
        self._agents = ["good", "adv"]   # agent names == roles for simplicity
        self._dist = {"good": 1.0, "adv": 1.0}  # 0..3
        self._cd   = {"good": 0.0, "adv": 0.0}
        self._wind = {"good": 0.0, "adv": 0.0}  # windup timer ticks
        self._dmg  = {}  # role -> (min,max)

    # ---- EnvAdapter API ----
    def reset(self, seed: int | None = None) -> Timestep:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._hp = self.hp0
        self._dist = {"good": 1.0, "adv": 1.0}
        self._cd   = {"good": 0.0, "adv": 0.0}
        self._wind = {"good": 0.0, "adv": 0.0}
        # slightly asymmetric damage bands per episode
        self._dmg = {
            "good": (42.0, 50.0),
            "adv":  (41.0, 49.0),
        }
        return self._obs_ts({})

    def step(self, actions: Dict[str, int]) -> Timestep:
        # resolve movement first
        for role, a in actions.items():
            if a == 1:  # step_in
                self._dist[role] = max(0.0, self._dist[role] - 1.0)
            elif a == 2:  # step_out
                self._dist[role] = min(3.0, self._dist[role] + 1.0)

        # tick down cooldowns/windups
        for r in self._agents:
            self._cd[r]   = max(0.0, self._cd[r] - 1.0)
            self._wind[r] = max(0.0, self._wind[r] - 1.0)

        # schedule attacks: if attack pressed and ready+in_range, set windup=1
        pending = []
        for role, a in actions.items():
            if a == 3 and self._cd[role] == 0.0 and self._dist[role] == 0.0:
                self._wind[role] = 1.0
            elif a == 4:
                self._wind[role] = 1.0  # fake: windup with no damage

        # decay creep HP (ambient dps)
        self._hp = max(0.0, self._hp - self.hp_decay)

        # apply damage for windups that just completed
        last_hit: str | None = None
        for role in self._agents:
            if self._wind[role] == 0.0:  # just completed this tick
                # check if attack was real: in range & cd was 0 last tick & not fake
                # (we can't perfectly know intent; assume if in range now, deal damage)
                if self._dist[role] == 0.0 and actions.get(role, 0) in (3,4):
                    if actions.get(role, 0) == 3:  # real attack
                        dmg = self._rng.uniform(*self._dmg[role])
                        prev_hp = self._hp
                        self._hp = max(0.0, self._hp - dmg)
                        self._cd[role] = self.base_cd
                        if prev_hp > 0.0 and self._hp == 0.0 and last_hit is None:
                            last_hit = role

        self._t += 1

        # rewards
        rew = {r: -0.005 for r in self._agents}
        if last_hit is not None:
            rew[last_hit] = 1.0
            other = "adv" if last_hit == "good" else "good"
            rew[other] = -1.0

        done_flag = (last_hit is not None) or (self._t >= self.max_steps) or (self._hp <= 0.0)
        dones = {r: done_flag for r in self._agents}

        return self._obs_ts(rew, dones)

    def close(self) -> None:
        pass

    def roles(self) -> Dict[str, str]:
        return {"good": "good", "adv": "adv"}

    def obs_dims(self) -> Dict[str, int]:
        return {"good": 12, "adv": 12}

    def n_actions(self) -> int:
        return 5

    def agent_names(self) -> List[str]:
        return ["good", "adv"]

    # ---- helpers ----
    def _obs(self, role: str) -> np.ndarray:
        opp = "adv" if role == "good" else "good"
        # normalize hp to 0..1
        hp_n = self._hp / max(1e-6, self.hp0)
        dmg_min, dmg_max = self._dmg[role]
        odmg_min, odmg_max = self._dmg[opp]
        lane_pressure = self._rng.random()
        vec = np.array([
            hp_n,
            self._dist[role], self._dist[opp],
            self._cd[role], self._cd[opp],
            dmg_min/100.0, dmg_max/100.0,
            odmg_min/100.0, odmg_max/100.0,
            1.0 if self._wind[role] > 0 else 0.0,
            1.0 if self._wind[opp]  > 0 else 0.0,
            lane_pressure,
        ], dtype=np.float32)
        return vec

    def _obs_ts(self, rewards: Dict[str,float] | None, dones: Dict[str,bool] | None = None) -> Timestep:
        obs = {r: self._obs(r) for r in self._agents}
        if rewards is None: rewards = {r: 0.0 for r in self._agents}
        if dones    is None: dones    = {r: False for r in self._agents}
        infos = {}
        return Timestep(obs=obs, rewards=rewards, dones=dones, infos=infos)