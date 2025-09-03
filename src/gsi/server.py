"""
Game State Integration HTTP server for Dota 2.

Receives POST requests with game state data and logs them as NDJSON.
"""

import json
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path


class GSIHandler(BaseHTTPRequestHandler):
    """HTTP handler for Dota 2 Game State Integration requests."""
    
    def __init__(self, *args, token=None, outdir=None, rotate_min=0, **kwargs):
        self.token = token
        self.outdir = Path(outdir) if outdir else Path("artifacts/gsi")
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rotate_min = rotate_min
        
        # Initialize session log file
        self._create_new_session()
        
        super().__init__(*args, **kwargs)
    
    def _create_new_session(self):
        """Create a new session log file and write session_start record."""
        timestamp = int(time.time())
        self.log_file = self.outdir / f"session-{timestamp}.ndjson"
        self.session_start = time.time()
        
        # Write session start record
        session_start = {
            "type": "session_start",
            "ts": self.session_start,
            "timestamp": timestamp
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(session_start, separators=(',', ':')) + '\n')
    
    def _check_rotation(self):
        """Check if log file should be rotated based on elapsed time."""
        if self.rotate_min > 0:
            elapsed_min = (time.time() - self.session_start) / 60.0
            if elapsed_min >= self.rotate_min:
                self._create_new_session()
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            # Health check endpoint
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            health_response = {
                "status": "ok",
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(health_response).encode('utf-8'))
        else:
            self.send_error(404, "Only GET /health is supported")
    
    def do_POST(self):
        """Handle POST requests with GSI data."""
        if self.path != "/":
            self.send_error(404, "Only POST / is supported")
            return
        
        # Check rotation before processing
        self._check_rotation()
        
        # Check authorization
        if not self._check_auth():
            self.send_error(401, "Unauthorized")
            return
        
        # Read request body
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            if not body:
                self.send_error(400, "Empty request body")
                return
            
            # Parse JSON to validate
            data = json.loads(body.decode('utf-8'))
            
        except (ValueError, json.JSONDecodeError) as e:
            self.send_error(400, f"Invalid JSON: {e}")
            return
        
        # Log the data
        self._log_data(body.decode('utf-8'))
        
        # Print debug info on first request
        if not hasattr(self.__class__, '_first_request_logged'):
            self._print_debug_info(data, len(body))
            self.__class__._first_request_logged = True
        
        # Send response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    
    def _check_auth(self):
        """Check Bearer token in Authorization header or JSON body."""
        if not self.token:
            return True  # No token required
        
        # Check Authorization header
        auth_header = self.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]  # Remove 'Bearer ' prefix
            if token == self.token:
                return True
        
        # Check JSON body for auth token (fallback)
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                # Read body without consuming the stream
                body = self.rfile.read(content_length)
                # Reset stream position for later reading
                import io
                self.rfile = io.BytesIO(body)
                
                data = json.loads(body.decode('utf-8'))
                auth_data = data.get('auth', {})
                if auth_data.get('token') == self.token:
                    return True
        except (ValueError, json.JSONDecodeError):
            pass
        
        return False
    
    def _log_data(self, body_str):
        """Append compact JSON line with timestamp and raw body."""
        log_entry = {
            "ts": time.time(),
            "data": json.loads(body_str)  # Parse to ensure valid JSON
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, separators=(',', ':')) + '\n')
    
    def _print_debug_info(self, data, body_size):
        """Print debug information for the first request."""
        print(f"📦 First GSI payload: {body_size} bytes")
        
        # Sample some interesting keys
        sample_keys = []
        
        def collect_keys(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        collect_keys(value, full_key)
                    else:
                        sample_keys.append(f"{full_key}={value}")
        
        collect_keys(data)
        
        # Print first few interesting keys
        interesting_keys = [k for k in sample_keys if any(x in k.lower() for x in 
                          ['game_time', 'health', 'level', 'gold', 'last_hits'])]
        
        if interesting_keys:
            print("🎮 Sample keys:", ', '.join(interesting_keys[:5]))
        elif sample_keys:
            print("📋 Sample keys:", ', '.join(sample_keys[:5]))
        
        print(f"📝 Logging to: {self.log_file}")
    
    def log_message(self, format, *args):
        """Override to reduce log spam."""
        # Only log errors, not every request
        if 'POST' not in format:
            super().log_message(format, *args)


def create_handler(token=None, outdir=None, rotate_min=0):
    """Create a handler class with bound token, output directory, and rotation."""
    class BoundGSIHandler(GSIHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, token=token, outdir=outdir, rotate_min=rotate_min, **kwargs)
    
    return BoundGSIHandler