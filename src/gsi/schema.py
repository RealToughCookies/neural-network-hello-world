"""
GSI data schemas and parsing utilities for Dota 2 game state.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime


@dataclass
class HeroState:
    """Hero information from GSI."""
    id: int
    name: str
    level: int
    xp: int
    alive: bool
    respawn_seconds: int
    buyback_cost: int
    buyback_cooldown: int
    health: int
    max_health: int
    health_percent: int
    mana: int
    max_mana: int
    mana_percent: int
    
    @classmethod
    def from_gsi(cls, data: Dict[str, Any]) -> "HeroState":
        """Parse hero state from GSI data."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", ""),
            level=data.get("level", 1),
            xp=data.get("xp", 0),
            alive=data.get("alive", True),
            respawn_seconds=data.get("respawn_seconds", 0),
            buyback_cost=data.get("buyback_cost", 0),
            buyback_cooldown=data.get("buyback_cooldown", 0),
            health=data.get("health", 100),
            max_health=data.get("max_health", 100),
            health_percent=data.get("health_percent", 100),
            mana=data.get("mana", 100),
            max_mana=data.get("max_mana", 100),
            mana_percent=data.get("mana_percent", 100),
        )


@dataclass
class ItemState:
    """Item information from GSI."""
    name: str
    purchaser: Optional[int]
    item_level: int
    can_cast: bool
    cooldown: int
    passive: bool
    
    @classmethod
    def from_gsi(cls, name: str, data: Dict[str, Any]) -> "ItemState":
        """Parse item state from GSI data."""
        return cls(
            name=name,
            purchaser=data.get("purchaser"),
            item_level=data.get("item_level", 1),
            can_cast=data.get("can_cast", False),
            cooldown=data.get("cooldown", 0),
            passive=data.get("passive", False),
        )


@dataclass
class CreepState:
    """Creep information from GSI."""
    name: str
    health: int
    max_health: int
    team: int  # 2 = radiant, 3 = dire
    gold_value: int
    xp_value: int
    
    @classmethod
    def from_gsi(cls, name: str, data: Dict[str, Any]) -> "CreepState":
        """Parse creep state from GSI data."""
        return cls(
            name=name,
            health=data.get("health", 100),
            max_health=data.get("max_health", 100),
            team=data.get("team", 2),
            gold_value=data.get("gold_value", 0),
            xp_value=data.get("xp_value", 0),
        )


@dataclass
class MapState:
    """Map and game information from GSI."""
    name: str
    matchid: str
    game_time: int
    clock_time: int
    daytime: bool
    nightstalker_night: bool
    radiant_score: int
    dire_score: int
    game_state: str
    paused: bool
    win_team: str
    
    @classmethod
    def from_gsi(cls, data: Dict[str, Any]) -> "MapState":
        """Parse map state from GSI data."""
        return cls(
            name=data.get("name", ""),
            matchid=data.get("matchid", ""),
            game_time=data.get("game_time", 0),
            clock_time=data.get("clock_time", 0),
            daytime=data.get("daytime", True),
            nightstalker_night=data.get("nightstalker_night", False),
            radiant_score=data.get("radiant_score", 0),
            dire_score=data.get("dire_score", 0),
            game_state=data.get("game_state", ""),
            paused=data.get("paused", False),
            win_team=data.get("win_team", ""),
        )


@dataclass
class PlayerState:
    """Player information from GSI."""
    steam_id: str
    name: str
    activity: str
    kills: int
    deaths: int
    assists: int
    last_hits: int
    denies: int
    kill_streak: int
    commands_issued: int
    kill_list: Dict[str, int]
    team_name: str
    gold: int
    gold_reliable: int
    gold_unreliable: int
    gold_from_hero_kills: int
    gold_from_creep_kills: int
    gold_from_income: int
    gold_from_shared: int
    gpm: int
    xpm: int
    
    @classmethod
    def from_gsi(cls, data: Dict[str, Any]) -> "PlayerState":
        """Parse player state from GSI data."""
        return cls(
            steam_id=data.get("steamid", ""),
            name=data.get("name", ""),
            activity=data.get("activity", ""),
            kills=data.get("kills", 0),
            deaths=data.get("deaths", 0),
            assists=data.get("assists", 0),
            last_hits=data.get("last_hits", 0),
            denies=data.get("denies", 0),
            kill_streak=data.get("kill_streak", 0),
            commands_issued=data.get("commands_issued", 0),
            kill_list=data.get("kill_list", {}),
            team_name=data.get("team_name", ""),
            gold=data.get("gold", 0),
            gold_reliable=data.get("gold_reliable", 0),
            gold_unreliable=data.get("gold_unreliable", 0),
            gold_from_hero_kills=data.get("gold_from_hero_kills", 0),
            gold_from_creep_kills=data.get("gold_from_creep_kills", 0),
            gold_from_income=data.get("gold_from_income", 0),
            gold_from_shared=data.get("gold_from_shared", 0),
            gpm=data.get("gpm", 0),
            xpm=data.get("xpm", 0),
        )


@dataclass
class GSISnapshot:
    """Complete GSI snapshot with timestamp."""
    timestamp: str
    provider: Dict[str, Any]
    map_state: Optional[MapState]
    player: Optional[PlayerState]  
    hero: Optional[HeroState]
    abilities: Dict[str, Dict[str, Any]]
    items: Dict[str, ItemState]
    wearables: Dict[str, Dict[str, Any]]
    buildings: Dict[str, Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_gsi_payload(cls, payload: Dict[str, Any]) -> "GSISnapshot":
        """Parse complete GSI snapshot from HTTP payload."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Parse nested structures
        map_data = payload.get("map", {})
        player_data = payload.get("player", {})
        hero_data = payload.get("hero", {})
        items_data = payload.get("items", {})
        
        # Convert items to ItemState objects
        items = {}
        for slot, item_data in items_data.items():
            if isinstance(item_data, dict) and "name" in item_data:
                items[slot] = ItemState.from_gsi(item_data["name"], item_data)
        
        return cls(
            timestamp=timestamp,
            provider=payload.get("provider", {}),
            map_state=MapState.from_gsi(map_data) if map_data else None,
            player=PlayerState.from_gsi(player_data) if player_data else None,
            hero=HeroState.from_gsi(hero_data) if hero_data else None,
            abilities=payload.get("abilities", {}),
            items=items,
            wearables=payload.get("wearables", {}),
            buildings=payload.get("buildings", {}),
        )


def parse_gsi_payload(payload: Dict[str, Any]) -> GSISnapshot:
    """Parse GSI HTTP payload into structured snapshot."""
    return GSISnapshot.from_gsi_payload(payload)