"""
GSI HTTP listener and data persistence for Dota 2.
"""

import json
import logging
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from .schema import GSISnapshot, parse_gsi_payload

logger = logging.getLogger(__name__)


class GSIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for GSI updates."""
    
    def __init__(self, *args, callback: Optional[Callable[[GSISnapshot], None]] = None, **kwargs):
        self.callback = callback
        super().__init__(*args, **kwargs)
    
    def do_POST(self):
        """Handle POST requests from Dota 2 GSI."""
        try:
            # Read the payload
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_response(400)
                self.end_headers()
                return
            
            raw_data = self.rfile.read(content_length)
            payload = json.loads(raw_data.decode('utf-8'))
            
            # Parse into structured data
            snapshot = parse_gsi_payload(payload)
            
            # Call the callback if provided
            if self.callback:
                self.callback(snapshot)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
            
        except Exception as e:
            logger.error(f"Error handling GSI request: {e}")
            self.send_response(500)
            self.end_headers()
    
    def do_GET(self):
        """Handle GET requests (health check)."""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'GSI Server Active')
    
    def log_message(self, format, *args):
        """Override to reduce noise."""
        pass  # Suppress default HTTP logging


class GSIClient:
    """GSI client with HTTP server and data persistence."""
    
    def __init__(self, host: str = "localhost", port: int = 3000, data_dir: str = "data/gsi"):
        self.host = host
        self.port = port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        self.callbacks = []
        
        # Rolling log files
        self.current_log_file: Optional[Path] = None
        self.log_rotation_interval = 3600  # 1 hour
        self.last_rotation = time.time()
        
        # Statistics
        self.snapshots_received = 0
        self.last_snapshot_time: Optional[str] = None
        
    def add_callback(self, callback: Callable[[GSISnapshot], None]):
        """Add callback function to be called on each GSI update."""
        self.callbacks.append(callback)
    
    def _handle_snapshot(self, snapshot: GSISnapshot):
        """Internal handler for GSI snapshots."""
        try:
            # Update statistics
            self.snapshots_received += 1
            self.last_snapshot_time = snapshot.timestamp
            
            # Save to rolling log
            self._save_snapshot(snapshot)
            
            # Call external callbacks
            for callback in self.callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(f"Error in GSI callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling GSI snapshot: {e}")
    
    def _save_snapshot(self, snapshot: GSISnapshot):
        """Save snapshot to rolling JSON log."""
        try:
            # Check if we need to rotate log file
            current_time = time.time()
            if (self.current_log_file is None or 
                current_time - self.last_rotation > self.log_rotation_interval):
                self._rotate_log_file()
            
            # Append to current log file
            if self.current_log_file:
                with open(self.current_log_file, "a") as f:
                    f.write(snapshot.to_json() + "\n")
                    
        except Exception as e:
            logger.error(f"Error saving GSI snapshot: {e}")
    
    def _rotate_log_file(self):
        """Create new log file for current time period."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.data_dir / f"gsi_log_{timestamp}.jsonl"
        self.last_rotation = time.time()
        logger.info(f"Rotated GSI log to: {self.current_log_file}")
    
    def start(self):
        """Start the GSI HTTP server."""
        if self.running:
            logger.warning("GSI client already running")
            return
        
        try:
            # Create HTTP server with custom handler
            def handler_factory(*args, **kwargs):
                return GSIHandler(*args, callback=self._handle_snapshot, **kwargs)
            
            self.server = HTTPServer((self.host, self.port), handler_factory)
            self.running = True
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            logger.info(f"GSI server started on {self.host}:{self.port}")
            logger.info(f"Data directory: {self.data_dir.resolve()}")
            
        except Exception as e:
            logger.error(f"Failed to start GSI server: {e}")
            self.running = False
            raise
    
    def stop(self):
        """Stop the GSI HTTP server."""
        if not self.running:
            return
        
        try:
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=5)
            
            self.running = False
            logger.info("GSI server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping GSI server: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "snapshots_received": self.snapshots_received,
            "last_snapshot_time": self.last_snapshot_time,
            "current_log_file": str(self.current_log_file) if self.current_log_file else None,
            "data_directory": str(self.data_dir.resolve()),
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def create_gsi_config(config_path: str = None, port: int = 3000) -> str:
    """Create GSI configuration file for Dota 2."""
    if config_path is None:
        # Default Dota 2 config location
        import os
        if os.name == 'nt':  # Windows
            config_path = os.path.expanduser("~/Documents/My Games/Dota 2/game/dota/cfg/gamestate_integration_coach.cfg")
        else:  # Linux/Mac
            config_path = os.path.expanduser("~/.local/share/Steam/userdata/*/570/local/cfg/gamestate_integration_coach.cfg")
    
    config_content = f'''
"Dota 2 Coach GSI Configuration"
{{
    "uri"               "http://localhost:{port}/"
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
    
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(config_content)
        return str(config_file.resolve())
    except Exception as e:
        logger.error(f"Failed to create GSI config: {e}")
        return config_content


if __name__ == "__main__":
    # Simple test/demo
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def test_callback(snapshot: GSISnapshot):
        """Test callback that prints basic info."""
        if snapshot.hero and snapshot.player:
            print(f"Hero: {snapshot.hero.name} | Level: {snapshot.hero.level} | "
                  f"CS: {snapshot.player.last_hits} | Denies: {snapshot.player.denies}")
    
    with GSIClient(port=3000) as client:
        client.add_callback(test_callback)
        print("GSI Client running. Start Dota 2 with -gamestateintegration launch option.")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(1)
                if client.snapshots_received > 0:
                    stats = client.get_stats()
                    print(f"\rSnapshots: {stats['snapshots_received']}, Last: {stats['last_snapshot_time']}", end="")
        except KeyboardInterrupt:
            print("\nStopping GSI client...")