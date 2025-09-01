# Dota 2 GSI Coaching System

A real-time coaching system for Dota 2 using Game State Integration (GSI) that provides last-hit timing, positioning advice, and farming efficiency feedback.

## Quick Start

### 1. Setup GSI Configuration
```bash
# Auto-setup GSI config file
python scripts/setup_gsi.py

# Or manually specify Dota 2 config directory
python scripts/setup_gsi.py --config-dir "/path/to/dota/cfg"
```

### 2. Start Live Coaching
```bash
# Basic rule-based coaching
python -m src.coach.live_shell --coach rules

# With text-to-speech
python -m src.coach.live_shell --coach rules --tts

# Using RL policy (requires trained model)
python -m src.coach.live_shell --coach rl --policy artifacts/rl_ckpts/best.pt
```

### 3. Launch Dota 2
Start Dota 2 with GSI enabled:
```bash
# Steam launch option
-gamestateintegration
```

## System Components

### GSI Client (`src/gsi/`)
- **HTTP Server**: Receives real-time game state from Dota 2
- **JSON Logging**: Persists game data for offline analysis
- **Schema Parsing**: Structured data extraction

### Coaching Engine (`src/coach/`)
- **Rule-based**: Heuristics for CS rate, deny rate, GPM
- **RL Integration**: ML-based attack timing suggestions
- **Live Shell**: Terminal output with optional TTS

### Offline Analysis (`scripts/`)
- **Replay System**: Process historical GSI logs
- **Performance Metrics**: CS@5min, CS@10min, coaching effectiveness
- **Benchmarking**: Compare against skill brackets

## Features

### Real-time Coaching
- **Last-hit Timing**: Attack now suggestions based on creep health
- **Farming Efficiency**: CS rate and GPM benchmarks
- **Deny Reminders**: Improve lane control
- **Item Timing**: Suggest farming items at key moments
- **Positioning**: Basic safety warnings (low HP, night time)

### Performance Tracking
- **CS Benchmarks**: Compare against skill brackets
  - Herald-Crusader: 25+ CS@5min
  - Archon-Legend: 30+ CS@5min  
  - Ancient+: 35+ CS@5min
- **Gold Efficiency**: GPM tracking and optimization
- **Deny Rate**: Target 10-20% for lane winners
- **Item Progression**: Core item timing analysis

### Coaching Modes
- **Rules-based**: Simple heuristics, works immediately
- **RL-enhanced**: Uses trained models for better timing
- **Quiet Mode**: Minimal output for experienced players
- **TTS Mode**: Audio coaching for hands-free operation

## Usage Examples

### Live Coaching Session
```bash
# Start with basic rule-based coaching
python -m src.coach.live_shell --coach rules

# Output example:
[12:34:56] [05:23] 🔥 LAST_HIT: CS behind benchmark: 18/38 (47%). Focus on last-hitting!
[12:35:02] [05:29] ⚠️ FARM: Low GPM: 284. Focus on farming efficiency!
[12:35:15] [05:42] ℹ️ POSITIONING: Nighttime - be more careful, vision is limited
```

### Offline Replay Analysis
```bash
# Analyze stored GSI logs
python scripts/replay_offline.py data/gsi_logs/match_123.jsonl --verbose

# Output coaching suggestions with timestamps
python scripts/replay_offline.py data/gsi_logs/match_123.jsonl \
  --coach rules --start-time 0 --end-time 600 --output analysis.json
```

### System Testing
```bash
# Run comprehensive system tests
python scripts/test_gsi_system.py

# Test GSI config generation
python scripts/setup_gsi.py --dry-run
```

## Configuration

### GSI Settings (gamestate_integration_coach.cfg)
```
"Dota 2 Coach GSI Configuration"
{
    "uri"         "http://localhost:3000/"
    "timeout"     "5.0"
    "buffer"      "0.1"
    "throttle"    "0.1" 
    "heartbeat"   "30.0"
    "data"
    {
        "provider"   "1"
        "map"        "1"
        "player"     "1" 
        "hero"       "1"
        "abilities"  "1"
        "items"      "1"
        "buildings"  "1"
    }
}
```

### Coaching Parameters
```python
# Suggestion rate limiting
suggestion_cooldown = 5.0  # seconds between similar suggestions
max_suggestions_per_type = 1  # per time window

# CS benchmarks (per 5 minutes)
cs_benchmarks = {
    "herald": 25,
    "crusader": 28, 
    "archon": 30,
    "legend": 32,
    "ancient": 35,
    "divine": 40,
    "immortal": 45
}

# GPM thresholds
gpm_thresholds = {
    "poor": 300,
    "decent": 400, 
    "good": 500,
    "excellent": 600
}
```

## Troubleshooting

### GSI Not Working
1. Check Dota 2 launch options include `-gamestateintegration`
2. Verify config file location and contents
3. Restart Dota 2 after config changes
4. Check port 3000 isn't blocked by firewall

### No Coaching Suggestions
1. Join a match (not main menu)
2. Wait for laning phase (suggestions minimal pre-game)
3. Check console for error messages
4. Verify hero/player data is being received

### Performance Issues
1. Reduce suggestion frequency with `--quiet` mode
2. Disable TTS if causing audio conflicts
3. Close other applications using port 3000
4. Check system resources during gameplay

## Development

### Adding New Coaching Rules
```python
# In src/coach/rules.py
def _analyze_new_metric(self, snapshot: GSISnapshot) -> List[CoachSuggestion]:
    suggestions = []
    
    if condition_met:
        suggestions.append(CoachSuggestion(
            type="new_metric",
            message="Your coaching message here",
            priority=2,  # 1=low, 2=medium, 3=high
            confidence=0.8,
            timestamp=snapshot.timestamp,
            context={"additional": "data"}
        ))
    
    return suggestions
```

### Custom RL Integration
```python
# In src/coach/rl_bridge.py
def gsi_to_rl_features(self, snapshot: GSISnapshot) -> np.ndarray:
    features = []
    # Extract features matching your training data
    features.extend([
        snapshot.hero.health_percent / 100.0,
        snapshot.player.last_hits / 100.0,
        # ... more features
    ])
    return np.array(features)
```

## Performance Metrics

### Success Metrics
- **CS Improvement**: 10-15% increase over baseline
- **Deny Rate**: 5+ percentage point improvement
- **GPM**: 50+ gold per minute improvement
- **Suggestion Accuracy**: 80%+ relevant suggestions
- **System Uptime**: 95%+ during matches

### Benchmark Comparisons
- **Dota Plus**: Item suggestions only
- **Manual Coaching**: 1-on-1, expensive, not real-time
- **This System**: Real-time, ML-enhanced, free & customizable

See `experiments/coach_mvp.md` for detailed performance analysis and experimental results.