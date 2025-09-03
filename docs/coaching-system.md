# Live Dota 2 Coaching System

Real-time coaching system that monitors GSI (Game State Integration) logs and provides voice feedback.

## Prerequisites

1. **Dota 2 with GSI enabled:**
   - Place `gamestate_integration_rtcoach.cfg` in: `~/Library/Application Support/Steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/`
   - Add Steam launch option: `-gamestateintegration`
   - Start GSI server: `python -m scripts.gsi_run --port 53100 --token secret --outdir artifacts/gsi`

2. **macOS with text-to-speech:**
   - Uses built-in `say` command
   - Test with: `say "Hello world"`

## Usage

### Live Coaching
Monitor GSI logs and speak coaching prompts in real-time:

```bash
# Basic usage
python -m scripts.coach_live --gsi-dir artifacts/gsi

# With custom voice and policy
python -m scripts.coach_live --gsi-dir artifacts/gsi --voice Alex --policy models/last_hit.pt
```

**Options:**
- `--gsi-dir`: Directory containing GSI session logs (default: `artifacts/gsi`)
- `--policy`: Path to PyTorch policy file for last-hit suggestions
- `--tau`: Attack probability threshold for last-hit prompts (default: 0.7)
- `--voice`: macOS voice name (e.g., 'Alex', 'Samantha')
- `--check-interval`: File polling interval in seconds (default: 1.0)

### Offline Evaluation
Analyze coaching effectiveness from logged GSI data:

```bash
# Evaluate latest session
LATEST=$(ls -t artifacts/gsi/session-*.ndjson | head -n1)
python -m scripts.coach_offline_eval --log "$LATEST"

# With policy evaluation
python -m scripts.coach_offline_eval --log "$LATEST" --policy models/last_hit.pt --tau 0.8
```

**Metrics:**
- CS@5min (creep score at 5 minutes)
- Denies@5min 
- Prompt precision/recall with Wilson 95% confidence intervals
- Per-prompt-type effectiveness analysis

## Coaching Rules

The system implements these heuristic rules:

| Condition | Prompt | Key |
|-----------|--------|-----|
| `hero.health_percent < 0.35` | "Back, low HP." | `retreat` |
| Any ability with `cooldown == 0` and `level > 0` | "Key ability up." | `abilities` |
| No TP scroll in inventory | "Buy TP." | `tp_scroll` |
| `player.gold >= 500` and no boots | "Consider Boots." | `boots` |
| Policy suggests attack (prob ≥ τ) | "Good time to last-hit." | `last_hit` |

## Rate Limiting & Quality Control

**Global Rate Limit:** Max 6 prompts per minute
**Per-Key Cooldown:** 6 seconds between same prompt types  
**Wilson Gating:** Suppresses prompts with confidence interval lower bound < 0.55

This prevents spam while learning which prompt types are actually helpful.

## File Structure

```
src/coach/
├── rules.py          # Pure coaching rule functions
├── prompt_bus.py     # Rate limiting + Wilson CI gating  
├── voice.py          # macOS text-to-speech wrapper
└── ...

scripts/
├── coach_live.py     # Real-time coaching monitor
├── coach_offline_eval.py # Post-game analysis
└── ...

artifacts/
├── gsi/              # GSI session logs from server
└── coach/            # Coaching statistics and state
```

## Example Session

1. **Start GSI server:**
   ```bash
   python -m scripts.gsi_run --port 53100 --token secret --outdir artifacts/gsi
   ```

2. **Start live coaching in another terminal:**
   ```bash
   python -m scripts.coach_live --voice Alex
   ```

3. **Play Dota 2** with GSI enabled - coaching prompts will be spoken automatically

4. **Analyze session afterwards:**
   ```bash
   LATEST=$(ls -t artifacts/gsi/session-*.ndjson | head -n1)
   python -m scripts.coach_offline_eval --log "$LATEST"
   ```

## Customization

- **Add new rules:** Extend `src/coach/rules.py` with new `should_*()` functions
- **Adjust rate limits:** Modify `PromptBus` parameters in `coach_live.py`
- **Change voice:** Use `--voice` parameter or modify `speak()` calls
- **Policy integration:** Load PyTorch models for last-hit timing suggestions

The system is designed to be easily extensible with additional coaching heuristics and machine learning components.