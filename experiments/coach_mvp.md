# Dota 2 Coach MVP - Performance Metrics & Evaluation

## Overview

This experiment evaluates the performance of our Dota 2 coaching system using Game State Integration (GSI) data. The system provides real-time last-hit timing, positioning advice, and farming efficiency feedback using both rule-based heuristics and reinforcement learning policies.

## System Components

### 1. GSI Data Intake (`src/gsi/`)
- **HTTP Listener**: Receives real-time game state from Dota 2 client
- **Schema Parsing**: Structured parsing of hero, items, creeps, map state
- **JSON Persistence**: Rolling log files with timestamps for offline analysis

### 2. Coaching System (`src/coach/`)
- **Rule-Based Coach**: Simple heuristics for CS rate, deny rate, GPM benchmarks
- **RL Bridge**: Integration with trained last-hit policy for attack timing
- **Live Shell**: Real-time terminal output with optional TTS support

### 3. Offline Analysis (`scripts/replay_offline.py`)
- **Log Replay**: Process historical GSI data for analysis
- **Metrics Calculation**: CS@5min, CS@10min, deny rates, suggestion accuracy
- **Performance Comparison**: Baseline vs. coached performance

## Key Metrics

### Last-Hit Performance
- **CS@5min**: Creep score at 5 minutes (benchmark: 38 for core roles)
- **CS@10min**: Creep score at 10 minutes (benchmark: 80 for core roles)
- **CS per minute**: Overall farming efficiency throughout game
- **Deny Rate**: Denies / (CS + Denies) ratio (target: 10-20%)

### Gold Efficiency
- **GPM**: Gold per minute (benchmarks: 300+ poor, 400+ decent, 500+ excellent)
- **Gold Sources**: Breakdown of creep kills vs. other sources
- **Item Timing**: Key farming items acquired by target times

### Coaching Effectiveness
- **Suggestion Rate**: Suggestions per minute by type and priority
- **Prediction Accuracy**: RL policy confidence vs. actual last-hit success
- **Response Time**: Time from suggestion to player action
- **Improvement Rate**: Performance delta with vs. without coaching

## Experimental Setup

### Data Collection
```bash
# Start GSI data collection
python -m src.gsi.client --port 3000 --data-dir data/gsi_logs

# Dota 2 launch with GSI enabled
steam://rungameid/570//-gamestateintegration
```

### Baseline Measurement (No Coaching)
```bash
# Collect 5-10 games without coaching active
# Focus on solo queue ranked matches for consistency
# Record: CS@5, CS@10, GPM, deny rate, item timings
```

### Coached Performance
```bash
# Rule-based coaching
python -m src.coach.live_shell --coach rules --tts

# RL-based coaching (requires trained policy)
python -m src.coach.live_shell --coach rl --policy artifacts/rl_ckpts/best.pt --tts

# Combined coaching
python -m src.coach.live_shell --coach combined --policy artifacts/rl_ckpts/best.pt
```

### Offline Analysis
```bash
# Analyze replay logs
python scripts/replay_offline.py data/gsi_logs/gsi_log_20241201_120000.jsonl \
  --coach rules --verbose --output results/analysis_rules.json

python scripts/replay_offline.py data/gsi_logs/gsi_log_20241201_120000.jsonl \
  --coach rl --policy artifacts/rl_ckpts/best.pt \
  --verbose --output results/analysis_rl.json
```

## Expected Results

### Hypothesis
1. **Rule-based coaching** should improve CS consistency by 10-15% through awareness
2. **RL coaching** should improve last-hit timing accuracy by 5-10% over rules
3. **Combined approach** should achieve best overall performance
4. **Greatest improvement** expected for players in Herald-Crusader brackets

### Success Criteria

#### Tier 1 Success (Basic Functionality)
- [ ] GSI system captures game state reliably (>95% uptime)
- [ ] Coach generates relevant suggestions (>80% accuracy by manual review)
- [ ] System runs without crashes for full game duration
- [ ] CS@5min improves by ≥5% over baseline

#### Tier 2 Success (Competitive Performance) 
- [ ] CS@5min improves by ≥10% over baseline
- [ ] Deny rate increases by ≥3 percentage points
- [ ] GPM increases by ≥50 over baseline
- [ ] RL suggestions show >70% confidence on successful last-hits

#### Tier 3 Success (Pro-Level Impact)
- [ ] CS@5min reaches 35+ consistently (90th percentile performance)
- [ ] System provides value for Ancient+ players
- [ ] Suggestion timing accuracy within 0.5 seconds of optimal
- [ ] Positive feedback from test players across skill ranges

## Benchmarks & Comparisons

### Professional Benchmarks
- **CS@5min**: 45-50 (pos 1/2), 35-40 (pos 3), 25-30 (pos 4/5)
- **CS@10min**: 90-100 (pos 1/2), 70-80 (pos 3), 50-60 (pos 4/5)
- **Deny Rate**: 15-25% for lane winners
- **GPM**: 600+ (pos 1), 500+ (pos 2/3), 350+ (pos 4/5)

### Existing Tools Comparison
- **Dota Plus Assistant**: Basic item/skill recommendations
- **Overwolf Apps**: Post-game analysis, limited real-time
- **Manual Coaching**: 1-on-1 sessions, expensive, not scalable

Our system advantages:
- Real-time feedback during gameplay
- ML-driven timing predictions
- Free and open source
- Customizable coaching intensity

## Test Scenarios

### Scenario 1: Laning Phase Focus (0-10 minutes)
- Hero: Position 1 carry (Anti-Mage, Phantom Assassin, etc.)
- Map: Safe lane farming
- Metrics: CS@5, CS@10, deny rate, deaths
- Expected: Most significant improvement here

### Scenario 2: Mid Game Efficiency (10-25 minutes)
- Focus: Jungle farming patterns, item timings
- Metrics: GPM, farm efficiency, map movements
- Expected: Moderate improvement with better routing

### Scenario 3: Mixed Role Testing
- Test across all positions (1-5)
- Metrics: Role-appropriate benchmarks
- Expected: Diminishing returns for support roles

## Data Analysis Plan

### Statistical Tests
- **Paired t-test**: Before/after coaching performance
- **Effect size**: Cohen's d for practical significance
- **Confidence intervals**: 95% CI for all key metrics
- **Multiple comparisons**: Bonferroni correction

### Visualization
- CS timeline graphs (coached vs. baseline)
- Suggestion frequency heatmaps by game time
- Performance distribution box plots
- Improvement correlation with initial skill level

### Reporting
- Automated metric calculation from replay logs
- Performance dashboard with key KPIs
- Player feedback surveys (subjective experience)
- Video highlights of coaching moments

## Implementation Status

### ✅ Completed
- [x] GSI client with JSON logging
- [x] Rule-based coaching heuristics
- [x] RL policy integration bridge
- [x] Live coaching shell with TTS
- [x] Offline replay analysis system
- [x] Metrics calculation framework

### 🔄 In Progress
- [ ] RL policy training on last-hit data
- [ ] Advanced creep health prediction
- [ ] Multi-role coaching customization
- [ ] Performance dashboard UI

### 📋 Planned
- [ ] A/B testing framework
- [ ] Player feedback collection system
- [ ] Advanced positioning analysis
- [ ] Team fight coaching integration
- [ ] Mobile companion app

## Usage Instructions

### Prerequisites
```bash
# Install dependencies (if using TTS)
pip install pyttsx3  # Optional for text-to-speech

# Configure Dota 2 GSI
python -c "from src.gsi.client import create_gsi_config; print(create_gsi_config())"
```

### Quick Start
```bash
# Start live coaching (terminal only)
python -m src.coach.live_shell --coach rules

# Start with TTS enabled
python -m src.coach.live_shell --coach rules --tts

# Analyze replay data
python scripts/replay_offline.py data/gsi_logs/latest.jsonl --verbose
```

### Advanced Usage
```bash
# Train custom RL policy (if you have training data)
python -m src.rl.ppo_selfplay_skeleton --train --env dota_last_hit

# Use trained policy in coaching
python -m src.coach.live_shell --coach rl --policy artifacts/rl_ckpts/best.pt
```

## Future Improvements

### Technical Enhancements
1. **Predictive Modeling**: Anticipate enemy behavior for better timing
2. **Computer Vision**: Direct screen analysis for more detailed state
3. **Audio Processing**: Parse in-game audio for additional context
4. **Mobile Integration**: Coach suggestions on second screen

### Coaching Features
1. **Adaptive Difficulty**: Adjust suggestions based on player skill
2. **Multi-Hero Specialization**: Hero-specific coaching profiles
3. **Team Coordination**: Multi-player coaching for team games
4. **Replay Review**: Post-game analysis with improvement suggestions

### Research Directions
1. **Behavioral Analysis**: Player response patterns to different suggestion types
2. **Skill Transfer**: How coaching improvements transfer across heroes/roles
3. **Long-term Retention**: Sustained improvement after coaching removal
4. **Cognitive Load**: Impact of real-time suggestions on gameplay performance