# Dota 2 Game State Integration Setup

This guide shows how to set up Dota 2 Game State Integration (GSI) for real-time coaching prompts.

## Overview

GSI allows external applications to receive live game data from Dota 2. Our system:
1. Runs an HTTP server to receive GSI data
2. Logs all game state events to NDJSON files  
3. Analyzes game state and generates coaching prompts
4. Rate-limits prompts to ≤6 per minute

## Quick Start

1. **Start the GSI server:**
   ```bash
   python -m scripts.gsi_run --port 53000 --token secret --outdir artifacts/gsi
   ```

2. **Install the GSI config file** (see below for details)

3. **Launch Dota 2** with `-gamestateintegration` launch option

4. **Play a bot match** for ~5 minutes to generate data

5. **Analyze the logs:**
   ```bash
   python -m scripts.gsi_replay --log "artifacts/gsi/session-*.ndjson" --max-rate 6
   ```

## GSI Configuration File

### File Location

The GSI config file must be placed in your Dota 2 installation at:
```
.../steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/
```

Create the `gamestate_integration` folder if it doesn't exist. Dota 2 uses the same GSI mechanism as CS:GO ([CS:GO GSI spec](https://developer.valvesoftware.com/wiki/Counter-Strike:_Global_Offensive_Game_State_Integration)).

**Windows:**
```
C:\Program Files (x86)\Steam\steamapps\common\dota 2 beta\game\dota\cfg\gamestate_integration\
```

**macOS:**
```
~/Library/Application Support/Steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/
```

**Linux:**
```
~/.steam/steam/steamapps/common/dota 2 beta/game/dota/cfg/gamestate_integration/
```

### Config File Content

Create a file named `gamestate_integration_rtcoach.cfg` with the following content:

```
"rtcoach"
{
  "uri" "http://127.0.0.1:53000/"
  "timeout" "5.0"
  "buffer"  "0.1"
  "throttle" "0.1"
  "heartbeat" "30.0"
  "data"
  {
    "provider"      "1"
    "map"           "1"  
    "player"        "1"
    "hero"          "1"
    "abilities"     "1"
    "items"         "1"
  }
  "auth"
  {
    "token" "secret"
  }
}
```

### Configuration Options

- **uri**: The endpoint where your GSI server is listening
- **timeout**: Maximum time to wait for response (seconds)
- **buffer**: Time to buffer events before sending (seconds)
- **throttle**: Minimum time between updates (seconds)  
- **heartbeat**: Send heartbeat every N seconds even if no changes
- **data**: Which game data sections to include (1 = enabled, 0 = disabled)
- **auth**: Authentication token (must match your server's `--token` parameter)

## Steam Launch Options

Add the launch option `-gamestateintegration` to Dota 2 in Steam ([GitHub guide](https://github.com/antonpup/Dota2GSI/blob/master/README.md)):

1. Right-click Dota 2 in your Steam library
2. Select "Properties" 
3. In "Launch Options", add: `-gamestateintegration`
4. Close the properties window

This enables Dota 2's Game State Integration system, which uses the same mechanism as CS:GO GSI.

## Testing the Setup

### 1. Server Test (No Dota)

Start the server and test with curl:

```bash
# Start server
python -m scripts.gsi_run --port 53000 --token secret --outdir artifacts/gsi

# Test in another terminal
curl -X POST http://127.0.0.1:53000/ \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer secret' \
  --data '{"provider":{"timestamp":123},"map":{"game_time":75}, "player":{"last_hits":9,"denies":1,"gpm":300,"xpm":400,"gold":450}, "hero":{"health_percent":0.30,"mana_percent":0.60,"level":5},"abilities":{"0":{"cooldown":0,"level":1},"1":{"cooldown":5.2,"level":1}},"items":{"0":{"name":"item_tango"},"1":{"name":"item_clarity"}}}'
```

You should see:
- Server logs the request
- A new NDJSON file appears in `artifacts/gsi/`
- Response: `{"status":"ok"}`

### 2. Live Dota Test

1. Start the server as above
2. Launch Dota 2 with the launch option
3. Start a bot match (Practice → Create Lobby → Fill Empty Slots with Bots)
4. Play for a few minutes
5. Check that `artifacts/gsi/session-*.ndjson` files are being created and updated

### 3. Replay Analysis Test

After collecting some game data:

```bash
python -m scripts.gsi_replay --log "artifacts/gsi/session-*.ndjson" --max-rate 6 --verbose
```

You should see coaching prompts like:
- `🎯 [75.0s] Back, low HP.` (when hero health < 35%)
- `🎯 [120.0s] Key ability up.` (when abilities are off cooldown)
- `🎯 [180.0s] Buy TP.` (when no TP scroll in inventory)  
- `🎯 [200.0s] Consider Boots.` (when gold ≥ 500 and no boots)

## Coaching Rules

The current system implements these basic rules:

| Condition | Prompt |
|-----------|--------|
| `hero.health_percent < 0.35` | "Back, low HP." |
| Any ability with `cooldown == 0` and `level > 0` | "Key ability up." |  
| No TP scroll in items | "Buy TP." |
| `player.gold >= 500` and no boots | "Consider Boots." |

Rate limiting ensures max 6 prompts per minute to avoid spam.

## Troubleshooting

### Server Won't Start
- **Port in use**: Kill existing process with `lsof -ti :53000 | xargs kill -9`
- **Permission denied**: Try a different port with `--port 53001`

### No Data from Dota  
- **Config file location**: Verify the GSI config file is in the right directory
- **Launch option**: Make sure `-gamestateintegration` is set in Steam
- **Firewall**: Check that port 53000 isn't blocked
- **Token mismatch**: Ensure server `--token` matches config file `auth.token`

### Authentication Errors
- **401 Unauthorized**: Check that the `auth.token` in your config file matches the server's `--token` parameter
- **No auth header**: Dota should automatically send the token based on the config file

### No Coaching Prompts
- **Empty logs**: Check that NDJSON files contain actual game data
- **Rate limiting**: Prompts are limited to 6 per minute
- **Conditions not met**: Verify your test scenario triggers the coaching rules

## Advanced Usage

### Custom Token
```bash
python -m scripts.gsi_run --port 53000 --token "my-secret-token-123" --outdir artifacts/gsi
```
Make sure to update the config file's `auth.token` to match.

### Different Port
```bash
python -m scripts.gsi_run --port 53001 --token secret --outdir artifacts/gsi  
```
Update the config file's `uri` to `http://127.0.0.1:53001/`.

### No Authentication
```bash  
python -m scripts.gsi_run --port 53000 --outdir artifacts/gsi
```
Remove the `auth` section from the config file entirely.

### Multiple Log Files
```bash
python -m scripts.gsi_replay --log "artifacts/gsi/session-*.ndjson" --max-rate 10
```

### Verbose Analysis
```bash
python -m scripts.gsi_replay --log "artifacts/gsi/session-*.ndjson" --verbose
```
Shows debug information and rate-limited prompts.

## Data Format

### NDJSON Log Format
```json
{"ts": 1756768890.208055, "data": {"map": {"game_time": 75}, "player": {"gold": 450}, "hero": {"health_percent": 0.30}}}
```

### GSI Data Structure  
The `data` field contains nested objects like:
- `map.game_time`: Current game time in seconds
- `player.gold`, `player.last_hits`, `player.gpm`: Player statistics
- `hero.health_percent`, `hero.level`: Hero state (percentages are 0.0-1.0)
- `abilities.0.cooldown`, `abilities.0.level`: Ability states by slot
- `items.0.name`: Item names by slot (e.g., "item_tpscroll", "item_boots")

## Next Steps

This basic system can be extended with:
- More sophisticated coaching rules
- Integration with RL policy outputs for last-hit timing
- Advanced game state analysis
- Custom prompt categories and priorities
- Web interface for real-time coaching display