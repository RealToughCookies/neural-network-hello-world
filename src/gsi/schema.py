from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

def _get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

@dataclass
class CreepState:
    """Placeholder for creep state data."""
    pass

@dataclass
class GSISnapshot:
    ts: float
    game_time: Optional[int]
    last_hits: Optional[int]
    denies: Optional[int]
    gpm: Optional[int]
    xpm: Optional[int]
    health_percent: Optional[float]
    mana_percent: Optional[float]
    hero_id: Optional[int]
    level: Optional[int]
    # keep abilities/items flexible for now; tests just need presence
    abilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    items: List[Dict[str, Any]] = field(default_factory=list)

def parse_gsi_payload(payload: Dict[str, Any], ts: Optional[float] = None) -> GSISnapshot:
    """Defensive parser for Dota 2 GSI payloads (map/player/hero/abilities/items)."""
    ts = float(ts if ts is not None else time.time())

    # Simple extractions (keys per common Dota GSI payloads)
    game_time = _get(payload, ["map", "game_time"])
    last_hits = _get(payload, ["player", "last_hits"])
    denies = _get(payload, ["player", "denies"])
    gpm = _get(payload, ["player", "gpm"])
    xpm = _get(payload, ["player", "xpm"])

    hero = payload.get("hero", {}) if isinstance(payload.get("hero"), dict) else {}
    health_percent = hero.get("health_percent")
    mana_percent = hero.get("mana_percent")
    level = hero.get("level")
    hero_id = hero.get("id") or hero.get("hero_id")

    # Abilities may be a dict of abilityN objects; keep raw but only include dicts
    abilities_raw = payload.get("abilities", {})
    abilities: Dict[str, Dict[str, Any]] = {}
    if isinstance(abilities_raw, dict):
        for k, v in abilities_raw.items():
            if isinstance(v, dict):
                abilities[k] = v

    # Items often live in a dict of slotN → {name, can_cast, cooldown, charges, ...}
    items_raw = payload.get("items", {})
    items: List[Dict[str, Any]] = []
    if isinstance(items_raw, dict):
        for v in items_raw.values():
            if isinstance(v, dict):
                items.append(v)
    elif isinstance(items_raw, list):
        items = [x for x in items_raw if isinstance(x, dict)]

    return GSISnapshot(
        ts=ts,
        game_time=game_time,
        last_hits=last_hits,
        denies=denies,
        gpm=gpm,
        xpm=xpm,
        health_percent=health_percent,
        mana_percent=mana_percent,
        hero_id=hero_id,
        level=level,
        abilities=abilities,
        items=items,
    )