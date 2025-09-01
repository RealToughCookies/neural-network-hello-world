#!/usr/bin/env python3
"""
Setup script for Dota 2 Game State Integration configuration.
"""

import argparse
import platform
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gsi.client import create_gsi_config


def find_dota_config_dir():
    """Find Dota 2 configuration directory."""
    system = platform.system().lower()
    
    if system == "windows":
        # Windows Steam locations
        possible_paths = [
            Path.home() / "Documents" / "My Games" / "Dota 2" / "game" / "dota" / "cfg",
            Path("C:/Program Files (x86)/Steam/userdata/*/570/local/cfg"),
            Path("C:/Program Files/Steam/userdata/*/570/local/cfg"),
        ]
    elif system == "darwin":  # macOS
        possible_paths = [
            Path.home() / "Library" / "Application Support" / "Steam" / "userdata" / "*" / "570" / "local" / "cfg",
            Path.home() / ".steam" / "userdata" / "*" / "570" / "local" / "cfg",
        ]
    else:  # Linux
        possible_paths = [
            Path.home() / ".steam" / "userdata" / "*" / "570" / "local" / "cfg",
            Path.home() / ".local" / "share" / "Steam" / "userdata" / "*" / "570" / "local" / "cfg",
        ]
    
    # Try to find existing config directories
    for path_pattern in possible_paths:
        if "*" in str(path_pattern):
            # Handle wildcard patterns
            parent = path_pattern.parent
            if parent.exists():
                for subdir in parent.iterdir():
                    if subdir.is_dir():
                        full_path = subdir / path_pattern.name
                        if full_path.exists():
                            return full_path
        else:
            if path_pattern.exists():
                return path_pattern
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Setup Dota 2 GSI configuration")
    parser.add_argument("--port", type=int, default=3000, help="GSI HTTP port")
    parser.add_argument("--config-dir", type=Path, help="Dota 2 config directory (auto-detect if not specified)")
    parser.add_argument("--dry-run", action="store_true", help="Show config content without writing file")
    
    args = parser.parse_args()
    
    print("🎮 Dota 2 GSI Configuration Setup")
    print("=" * 40)
    
    # Find config directory
    if args.config_dir:
        config_dir = args.config_dir
    else:
        print("🔍 Auto-detecting Dota 2 config directory...")
        config_dir = find_dota_config_dir()
        
        if config_dir:
            print(f"✅ Found: {config_dir}")
        else:
            print("❌ Could not auto-detect Dota 2 config directory")
            print("\nPossible locations:")
            system = platform.system().lower()
            
            if system == "windows":
                print("  - Documents/My Games/Dota 2/game/dota/cfg/")
                print("  - Steam/userdata/[USER_ID]/570/local/cfg/")
            elif system == "darwin":
                print("  - ~/Library/Application Support/Steam/userdata/[USER_ID]/570/local/cfg/")
            else:
                print("  - ~/.steam/userdata/[USER_ID]/570/local/cfg/")
                print("  - ~/.local/share/Steam/userdata/[USER_ID]/570/local/cfg/")
            
            print(f"\nManually specify with: --config-dir <path>")
            return 1
    
    # Generate config content
    config_content = f'''
"Dota 2 Coach GSI Configuration"
{{
    "uri"               "http://localhost:{args.port}/"
    "timeout"           "5.0"
    "buffer"            "0.1"
    "throttle"          "0.1"
    "heartbeat"         "30.0"
    "data"
    {{
        "provider"      "1"
        "map"           "1"
        "player"        "1"
        "hero"          "1"
        "abilities"     "1"
        "items"         "1"
        "buildings"     "1"
    }}
}}
'''.strip()
    
    if args.dry_run:
        print("\n📄 GSI Configuration content:")
        print(config_content)
        return 0
    
    # Write config file
    config_file = config_dir / "gamestate_integration_coach.cfg"
    
    try:
        # Create directory if it doesn't exist
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Write config file
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ GSI configuration written to:")
        print(f"   {config_file}")
        
        print("\n🚀 Next steps:")
        print("1. Restart Dota 2 (if already running)")
        print("2. Launch Dota 2 with: -gamestateintegration")
        print("3. Start coaching with:")
        print(f"   python -m src.coach.live_shell --coach rules --port {args.port}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to write config file: {e}")
        print("\n📄 Manual setup - save this as gamestate_integration_coach.cfg:")
        print(config_content)
        return 1


if __name__ == "__main__":
    sys.exit(main())