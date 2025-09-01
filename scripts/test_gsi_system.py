#!/usr/bin/env python3
"""
Test script for GSI system - creates mock data and tests all components.
"""

import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gsi.schema import GSISnapshot, parse_gsi_payload
from src.coach.rules import ComboCoach
from src.coach.rl_bridge import CombinedRLCoach


def create_mock_gsi_data(game_time: int = 300, cs_count: int = 25) -> dict:
    """Create mock GSI data for testing."""
    return {
        "provider": {
            "name": "Dota 2",
            "appid": 570,
            "version": 48,
            "timestamp": int(time.time())
        },
        "map": {
            "name": "dota",
            "matchid": "test_match_123",
            "game_time": game_time,
            "clock_time": game_time,
            "daytime": True,
            "nightstalker_night": False,
            "radiant_score": 5,
            "dire_score": 3,
            "game_state": "DOTA_GAMERULES_STATE_GAME_IN_PROGRESS",
            "paused": False,
            "win_team": ""
        },
        "player": {
            "steamid": "76561198000000000",
            "name": "TestPlayer",
            "activity": "playing",
            "kills": 3,
            "deaths": 1,
            "assists": 2,
            "last_hits": cs_count,
            "denies": int(cs_count * 0.2),
            "kill_streak": 2,
            "commands_issued": 1500,
            "kill_list": {},
            "team_name": "radiant",
            "gold": 2500,
            "gold_reliable": 1200,
            "gold_unreliable": 1300,
            "gold_from_hero_kills": 800,
            "gold_from_creep_kills": 1500,
            "gold_from_income": 200,
            "gold_from_shared": 0,
            "gpm": int(2500 / max(game_time / 60, 1)),
            "xpm": int(3000 / max(game_time / 60, 1))
        },
        "hero": {
            "id": 1,
            "name": "npc_dota_hero_antimage",
            "level": 6,
            "xp": 2800,
            "alive": True,
            "respawn_seconds": 0,
            "buyback_cost": 0,
            "buyback_cooldown": 0,
            "health": 580,
            "max_health": 580,
            "health_percent": 100,
            "mana": 220,
            "max_mana": 220,
            "mana_percent": 100
        },
        "abilities": {
            "ability0": {
                "name": "antimage_mana_break",
                "level": 2,
                "can_cast": True,
                "passive": True,
                "cooldown": 0
            }
        },
        "items": {
            "slot0": {
                "name": "item_quelling_blade",
                "purchaser": 0,
                "item_level": 1,
                "can_cast": True,
                "cooldown": 0,
                "passive": False
            },
            "slot1": {
                "name": "item_wraith_band",
                "purchaser": 0,
                "item_level": 1,
                "can_cast": False,
                "cooldown": 0,
                "passive": True
            }
        },
        "buildings": {}
    }


def test_gsi_parsing():
    """Test GSI data parsing."""
    print("Testing GSI parsing...")
    
    mock_data = create_mock_gsi_data()
    snapshot = parse_gsi_payload(mock_data)
    
    assert snapshot.hero is not None
    assert snapshot.player is not None
    assert snapshot.map_state is not None
    assert snapshot.hero.name == "npc_dota_hero_antimage"
    assert snapshot.player.last_hits == 25
    
    print("✅ GSI parsing works")


def test_rule_coaching():
    """Test rule-based coaching."""
    print("Testing rule-based coaching...")
    
    coach = ComboCoach()
    
    # Test with low CS scenario
    low_cs_data = create_mock_gsi_data(game_time=600, cs_count=20)  # 5 minutes, 20 CS
    snapshot = parse_gsi_payload(low_cs_data)
    suggestions = coach.analyze(snapshot)
    
    print(f"Low CS suggestions: {len(suggestions)}")
    for suggestion in suggestions:
        print(f"  - {suggestion.type}: {suggestion.message}")
    
    # Should have suggestions about low CS
    cs_suggestions = [s for s in suggestions if s.type == "last_hit"]
    assert len(cs_suggestions) > 0, "Should have CS suggestions for poor performance"
    
    # Test with good CS scenario
    good_cs_data = create_mock_gsi_data(game_time=600, cs_count=45)  # 5 minutes, 45 CS
    snapshot = parse_gsi_payload(good_cs_data)
    suggestions = coach.analyze(snapshot)
    
    print(f"Good CS suggestions: {len(suggestions)}")
    for suggestion in suggestions:
        print(f"  - {suggestion.type}: {suggestion.message}")
    
    print("✅ Rule-based coaching works")


def test_rl_bridge():
    """Test RL bridge (without actual policy)."""
    print("Testing RL bridge...")
    
    # Test without policy (should handle gracefully)
    coach = CombinedRLCoach(policy_path=None)
    mock_data = create_mock_gsi_data()
    snapshot = parse_gsi_payload(mock_data)
    
    suggestions = coach.analyze(snapshot)
    print(f"RL suggestions (no policy): {len(suggestions)}")
    
    # Should not crash, but may have no suggestions
    print("✅ RL bridge handles missing policy gracefully")


def test_metrics_calculation():
    """Test metrics calculation from mock data."""
    print("Testing metrics calculation...")
    
    from scripts.replay_offline import CoachingMetrics
    
    metrics = CoachingMetrics()
    
    # Simulate game progression
    for game_time in range(0, 1200, 60):  # 20 minutes, every minute
        cs_count = min(80, int(game_time / 60 * 4))  # ~4 CS per minute
        mock_data = create_mock_gsi_data(game_time=game_time, cs_count=cs_count)
        snapshot = parse_gsi_payload(mock_data)
        metrics.update_snapshot_metrics(snapshot)
    
    final_metrics = metrics.calculate_final_metrics()
    
    print(f"Duration: {final_metrics['duration_minutes']:.1f} minutes")
    print(f"Final CS: {final_metrics['cs_total']}")
    print(f"CS/min: {final_metrics['cs_per_minute']:.1f}")
    
    if 'cs_at_5min' in final_metrics:
        print(f"CS@5min: {final_metrics['cs_at_5min']:.0f}")
    
    assert final_metrics['duration_minutes'] > 15
    assert final_metrics['cs_total'] > 0
    
    print("✅ Metrics calculation works")


def simulate_coaching_session():
    """Simulate a coaching session with mock data."""
    print("\nSimulating coaching session...")
    
    coach = ComboCoach()
    
    # Simulate 10 minutes of gameplay
    for minute in range(1, 11):
        game_time = minute * 60
        
        # Simulate realistic CS progression
        if minute <= 5:
            cs_count = minute * 6  # 6 CS per minute early
        else:
            cs_count = 30 + (minute - 5) * 4  # Slower after 5 minutes
        
        mock_data = create_mock_gsi_data(game_time=game_time, cs_count=cs_count)
        snapshot = parse_gsi_payload(mock_data)
        suggestions = coach.analyze(snapshot)
        
        if suggestions:
            print(f"\nMinute {minute} ({cs_count} CS):")
            for suggestion in suggestions[:2]:  # Show top 2 suggestions
                priority = "🔥" if suggestion.priority == 3 else "⚠️" if suggestion.priority == 2 else "ℹ️"
                print(f"  {priority} {suggestion.message}")
    
    print("\n✅ Coaching session simulation complete")


def main():
    """Run all tests."""
    print("🎮 Testing GSI Coaching System\n")
    
    try:
        test_gsi_parsing()
        print()
        
        test_rule_coaching() 
        print()
        
        test_rl_bridge()
        print()
        
        test_metrics_calculation()
        
        simulate_coaching_session()
        
        print("\n🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Start Dota 2 with -gamestateintegration launch option")
        print("2. Create GSI config file (see experiments/coach_mvp.md)")
        print("3. Run: python -m src.coach.live_shell --coach rules")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())