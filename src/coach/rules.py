"""
Simple rule-based coaching heuristics for Dota 2.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from ..gsi.schema import GSISnapshot

logger = logging.getLogger(__name__)


@dataclass
class CoachSuggestion:
    """A coaching suggestion with priority and timing."""
    type: str  # "last_hit", "deny", "positioning", "farm", "item"
    message: str
    priority: int  # 1 = low, 2 = medium, 3 = high
    confidence: float  # 0.0 to 1.0
    timestamp: str
    context: Dict[str, Any]  # Additional context data


class SimpleLastHitCoach:
    """Rule-based last-hit coaching using GSI data."""
    
    def __init__(self):
        self.last_cs_count = 0
        self.last_deny_count = 0
        self.last_game_time = 0
        self.recent_suggestions = []  # Rate limiting
        self.suggestion_cooldown = 5.0  # seconds
        
    def analyze(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
        """Analyze snapshot and generate coaching suggestions."""
        suggestions = []
        
        if not snapshot.player or not snapshot.map_state:
            return suggestions
        
        game_time = snapshot.map_state.game_time
        cs_count = snapshot.player.last_hits
        deny_count = snapshot.player.denies
        
        # CS rate analysis
        suggestions.extend(self._analyze_cs_rate(snapshot, game_time, cs_count))
        
        # Deny rate analysis  
        suggestions.extend(self._analyze_deny_rate(snapshot, game_time, deny_count))
        
        # Gold efficiency
        suggestions.extend(self._analyze_gold_efficiency(snapshot))
        
        # Item timing
        suggestions.extend(self._analyze_item_timing(snapshot, game_time))
        
        # Update tracking
        self.last_cs_count = cs_count
        self.last_deny_count = deny_count
        self.last_game_time = game_time
        
        return suggestions
    
    def _analyze_cs_rate(self, snapshot: GSISnapshot, game_time: int, cs_count: int) -> List[CoachSuggestion]:
        """Analyze creep score rate and provide feedback."""
        suggestions = []
        
        if game_time <= 0:
            return suggestions
        
        # Expected CS benchmarks (rough guidelines)
        minutes = game_time / 60
        if minutes >= 5:
            expected_cs_5min = 38  # Good CS at 5 minutes for core
            expected_cs_10min = 80  # Good CS at 10 minutes
            
            if minutes <= 5:
                expected = expected_cs_5min * (minutes / 5)
            elif minutes <= 10:
                expected = expected_cs_5min + (expected_cs_10min - expected_cs_5min) * ((minutes - 5) / 5)
            else:
                expected = expected_cs_10min + (minutes - 10) * 7  # ~7 CS per minute after 10min
            
            cs_ratio = cs_count / max(expected, 1)
            
            if cs_ratio < 0.7:
                suggestions.append(CoachSuggestion(
                    type="last_hit",
                    message=f"CS behind benchmark: {cs_count}/{expected:.0f} ({cs_ratio:.1%}). Focus on last-hitting!",
                    priority=3,
                    confidence=0.8,
                    timestamp=snapshot.timestamp,
                    context={"cs_count": cs_count, "expected": expected, "ratio": cs_ratio}
                ))
            elif cs_ratio > 1.2:
                suggestions.append(CoachSuggestion(
                    type="last_hit", 
                    message=f"Excellent CS! {cs_count}/{expected:.0f} ({cs_ratio:.1%})",
                    priority=1,
                    confidence=0.9,
                    timestamp=snapshot.timestamp,
                    context={"cs_count": cs_count, "expected": expected, "ratio": cs_ratio}
                ))
        
        return suggestions
    
    def _analyze_deny_rate(self, snapshot: GSISnapshot, game_time: int, deny_count: int) -> List[CoachSuggestion]:
        """Analyze deny rate and provide feedback."""
        suggestions = []
        
        if game_time <= 0:
            return suggestions
        
        # Deny ratio vs last hits (good players aim for ~10-20% deny rate)
        cs_count = snapshot.player.last_hits
        if cs_count > 10:  # Only analyze if we have reasonable sample
            deny_ratio = deny_count / max(cs_count, 1)
            
            if deny_ratio < 0.05:  # Less than 5% denies
                suggestions.append(CoachSuggestion(
                    type="deny",
                    message=f"Low deny rate: {deny_count}/{cs_count} ({deny_ratio:.1%}). Try to deny enemy creeps!",
                    priority=2,
                    confidence=0.7,
                    timestamp=snapshot.timestamp,
                    context={"denies": deny_count, "cs": cs_count, "ratio": deny_ratio}
                ))
            elif deny_ratio > 0.25:  # More than 25% - very good
                suggestions.append(CoachSuggestion(
                    type="deny",
                    message=f"Great deny rate! {deny_count}/{cs_count} ({deny_ratio:.1%})",
                    priority=1,
                    confidence=0.8,
                    timestamp=snapshot.timestamp,
                    context={"denies": deny_count, "cs": cs_count, "ratio": deny_ratio}
                ))
        
        return suggestions
    
    def _analyze_gold_efficiency(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
        """Analyze gold per minute and efficiency."""
        suggestions = []
        
        if not snapshot.player or not snapshot.map_state:
            return suggestions
        
        gpm = snapshot.player.gpm
        game_time = snapshot.map_state.game_time
        minutes = game_time / 60
        
        # GPM benchmarks (rough)
        if minutes >= 5:
            if gpm < 300:
                suggestions.append(CoachSuggestion(
                    type="farm",
                    message=f"Low GPM: {gpm}. Focus on farming efficiency!",
                    priority=3,
                    confidence=0.7,
                    timestamp=snapshot.timestamp,
                    context={"gpm": gpm, "time": minutes}
                ))
            elif gpm > 500:
                suggestions.append(CoachSuggestion(
                    type="farm",
                    message=f"Excellent GPM: {gpm}!",
                    priority=1,
                    confidence=0.8,
                    timestamp=snapshot.timestamp,
                    context={"gpm": gpm, "time": minutes}
                ))
        
        return suggestions
    
    def _analyze_item_timing(self, snapshot: GSISnapshot, game_time: int) -> List[CoachSuggestion]:
        """Analyze item progression timing."""
        suggestions = []
        
        if not snapshot.items or not snapshot.hero:
            return suggestions
        
        minutes = game_time / 60
        hero_name = snapshot.hero.name.lower()
        
        # Check for key farming items
        has_farming_item = any(
            item.name in ["item_hand_of_midas", "item_maelstrom", "item_battle_fury", "item_radiance"]
            for item in snapshot.items.values()
        )
        
        # Suggest farming items for cores if none present by 15 minutes
        if minutes > 15 and not has_farming_item:
            if any(core_indicator in hero_name for core_indicator in ["carry", "mid", "core"]):
                suggestions.append(CoachSuggestion(
                    type="item",
                    message="Consider farming items like Maelstrom or Battle Fury for faster farm",
                    priority=2,
                    confidence=0.6,
                    timestamp=snapshot.timestamp,
                    context={"time": minutes, "hero": hero_name}
                ))
        
        return suggestions


class PositionCoach:
    """Rule-based positioning advice."""
    
    def __init__(self):
        pass
    
    def analyze(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
        """Analyze positioning and provide suggestions."""
        suggestions = []
        
        if not snapshot.hero or not snapshot.map_state:
            return suggestions
        
        # Basic positioning rules based on game state
        game_time = snapshot.map_state.game_time
        minutes = game_time / 60
        hero_hp_pct = snapshot.hero.health_percent
        
        # Low HP warning
        if hero_hp_pct < 30 and snapshot.hero.alive:
            suggestions.append(CoachSuggestion(
                type="positioning",
                message=f"Low HP ({hero_hp_pct}%)! Consider backing or healing",
                priority=3,
                confidence=0.9,
                timestamp=snapshot.timestamp,
                context={"hp_percent": hero_hp_pct}
            ))
        
        # Night/day positioning advice
        if not snapshot.map_state.daytime and minutes < 20:
            suggestions.append(CoachSuggestion(
                type="positioning", 
                message="Nighttime - be more careful, vision is limited",
                priority=2,
                confidence=0.7,
                timestamp=snapshot.timestamp,
                context={"daytime": False}
            ))
        
        return suggestions


class ComboCoach:
    """Combined coaching system."""
    
    def __init__(self):
        self.lasthit_coach = SimpleLastHitCoach()
        self.position_coach = PositionCoach()
        self.last_analysis_time = 0
        self.analysis_interval = 2.0  # Analyze every 2 seconds
    
    def analyze(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
        """Run all coaching analysis."""
        import time
        current_time = time.time()
        
        # Rate limit analysis
        if current_time - self.last_analysis_time < self.analysis_interval:
            return []
        
        suggestions = []
        
        try:
            suggestions.extend(self.lasthit_coach.analyze(snapshot))
            suggestions.extend(self.position_coach.analyze(snapshot))
        except Exception as e:
            logger.error(f"Error in coaching analysis: {e}")
        
        self.last_analysis_time = current_time
        
        # Sort by priority (high to low)
        suggestions.sort(key=lambda x: x.priority, reverse=True)
        
        return suggestions