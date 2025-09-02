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

from typing import Union, Iterable

PayloadLike = Union[dict, GSISnapshot]

def _as_payload(p: PayloadLike) -> dict:
    if isinstance(p, GSISnapshot):
        # Build a dict view sufficient for the helpers
        return {
            "hero": {
                "health_percent": p.health_percent,
                "mana_percent": p.mana_percent,
                "level": p.level,
                "id": p.hero_id,
            },
            "abilities": p.abilities,
            "items": {str(i): it for i, it in enumerate(p.items or [])},
        }
    return p if isinstance(p, dict) else {}

def get_health_percent(payload: PayloadLike) -> float | None:
    hero = _as_payload(payload).get("hero") or {}
    return hero.get("health_percent")

def get_mana_percent(payload: PayloadLike) -> float | None:
    hero = _as_payload(payload).get("hero") or {}
    return hero.get("mana_percent")

def any_ability_ready(payload: PayloadLike) -> bool:
    abilities = _as_payload(payload).get("abilities") or {}
    if not isinstance(abilities, dict):
        return False
    for v in abilities.values():
        if not isinstance(v, dict):
            continue
        lvl = int(v.get("level", 0) or 0)
        cd  = float(v.get("cooldown", 0) or 0.0)
        if lvl > 0 and cd <= 0:
            return True
    return False

def has_tp_scroll(payload: PayloadLike) -> bool:
    # TP Scroll or Boots of Travel count as a TP option
    names = _item_names(payload)
    return any(n in names for n in (
        "item_tpscroll", "item_travel_boots", "item_travel_boots_2"
    ))

def has_boots(payload: PayloadLike) -> bool:
    names = _item_names(payload)
    _BOOT_TOKENS = ("boots", "phase_boots", "power_treads", "tranquil_boots",
                   "arcane_boots", "guardian_greaves")
    return any(("item_" + tok) in names or tok in names for tok in _BOOT_TOKENS)

def _item_names(payload: PayloadLike) -> set[str]:
    items_raw = _as_payload(payload).get("items") or {}
    names: set[str] = set()
    if isinstance(items_raw, dict):
        for it in items_raw.values():
            if isinstance(it, dict) and isinstance(it.get("name"), str):
                names.add(it["name"])
    elif isinstance(items_raw, list):
        for it in items_raw:
            if isinstance(it, dict) and isinstance(it.get("name"), str):
                names.add(it["name"])
    return names