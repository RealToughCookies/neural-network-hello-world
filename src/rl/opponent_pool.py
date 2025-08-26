from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json, time, hashlib, math, os
from pathlib import Path

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _hash_path(p: str) -> str:
    return hashlib.sha1(p.encode()).hexdigest()[:10]

@dataclass
class OppEntry:
    id: str
    ckpt_path: str
    role: str
    step: int
    elo: float = 1000.0
    games: int = 0
    wins: int = 0
    losses: int = 0
    ema_exploit: float = 0.0
    added: str = ""

class OpponentPool:
    def __init__(self, env: str, obs_dims: Dict[str,int], schema: str = "v1-elo-pool"):
        self.schema = schema
        self.env = env
        self.obs_dims = dict(obs_dims)
        self.created = _now_iso()
        self.entries: Dict[str, OppEntry] = {}

    # ---------- I/O ----------
    @classmethod
    def load(cls, path: str) -> "OpponentPool":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        obj = json.loads(p.read_text())
        assert obj.get("schema") == "v1-elo-pool", "Unsupported pool schema"
        pool = cls(env=obj["env"], obs_dims=obj["obs_dims"], schema=obj["schema"])
        pool.created = obj.get("created") or pool.created
        for e in obj.get("entries", []):
            oe = OppEntry(**e)
            pool.entries[oe.id] = oe
        return pool

    def save(self, path: str) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "schema": self.schema,
            "env": self.env,
            "obs_dims": self.obs_dims,
            "created": self.created,
            "entries": [asdict(e) for e in self.entries.values()]
        }
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(p)

    # ---------- core ----------
    def add_entry(self, ckpt_path: str, role: str, step: int, init_elo: float = 1000.0) -> str:
        eid = f"{Path(ckpt_path).name}@step={step}#{_hash_path(ckpt_path)}"
        if eid in self.entries: return eid
        self.entries[eid] = OppEntry(
            id=eid, ckpt_path=ckpt_path, role=role, step=step,
            elo=init_elo, games=0, wins=0, losses=0, ema_exploit=0.0, added=_now_iso()
        )
        return eid

    def list_by_elo(self) -> List[OppEntry]:
        return sorted(self.entries.values(), key=lambda e: e.elo, reverse=True)

    def prune(self, max_entries: int = 64):
        items = self.list_by_elo()
        for e in items[max_entries:]:
            self.entries.pop(e.id, None)

    def sample(self, n: int, temp: float = 0.7, min_games: int = 5,
               min_selfplay_frac: float = 0.10, selfplay_role: str = "good") -> List[Tuple[str, OppEntry]]:
        """
        Return (kind, entry) where kind in {"self","pool"}; ensure >= selfplay_frac.
        """
        out: List[Tuple[str, OppEntry]] = []
        num_self = max(0, int(round(n * min_selfplay_frac)))
        # self-play slots
        for _ in range(num_self): out.append(("self", None))
        # pool slots
        pool = [e for e in self.entries.values() if e.games >= min_games]
        if not pool:
            out.extend([("self", None)] * (n - len(out)))
            return out[:n]
        # softmax over elo (higher elo → more probable), temperature temp
        elos = [e.elo for e in pool]
        m = max(elos)
        logits = [ (x - m) / max(1e-6, temp) for x in elos ]
        ws = [ math.exp(z) for z in logits ]
        s = sum(ws)
        ps = [ w/s for w in ws ]
        import random
        for _ in range(n - len(out)):
            e = random.choices(pool, weights=ps, k=1)[0]
            out.append(("pool", e))
        return out

    def record_result(self, opp_id: str, learner_win: bool, ema_decay: float = 0.97, K: float = 16.0):
        e = self.entries.get(opp_id)
        if e is None: 
            return
        # learner vs opponent model rating; we only update opponent's Elo (learner is on-line)
        # estimate learner rating as the mean of top opponents (or 1000 if unknown)
        top = self.list_by_elo()[:5]
        learner_elo = sum([x.elo for x in top]) / max(1, len(top)) if top else 1000.0
        expected_opp = 1.0 / (1.0 + 10 ** ((learner_elo - e.elo) / 400.0))
        result_for_opp = 0.0 if learner_win else 1.0
        # K factor anneal with games
        K_eff = K / math.sqrt(max(1.0, e.games + 1))
        e.elo = float(e.elo + K_eff * (result_for_opp - expected_opp))
        e.games += 1
        if learner_win: e.losses += 1
        else: e.wins += 1
        # exploitability: higher when learner beats it
        e.ema_exploit = float(ema_decay * e.ema_exploit + (1 - ema_decay) * (1.0 if learner_win else 0.0))