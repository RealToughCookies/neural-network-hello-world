"""
Tests for GSI server functionality.
"""

import json
import time
import tempfile
from pathlib import Path
from http.client import HTTPConnection
from threading import Thread
from http.server import HTTPServer
import pytest

from src.gsi.server import create_handler


class TestGSIServer:
    """Tests for GSI server functionality."""

    def test_health_endpoint(self):
        """Test GET /health returns JSON 200 response."""
        # Create a temporary directory for logs
        with tempfile.TemporaryDirectory() as tmpdir:
            handler_class = create_handler(token=None, outdir=tmpdir, rotate_min=0)
            
            # Start server on random port
            server = HTTPServer(("127.0.0.1", 0), handler_class)
            server_thread = Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            try:
                # Get actual bound port
                _, port = server.server_address
                
                # Test health endpoint
                conn = HTTPConnection("127.0.0.1", port)
                conn.request("GET", "/health")
                response = conn.getresponse()
                
                assert response.status == 200
                assert response.getheader("Content-Type") == "application/json"
                
                body = response.read().decode('utf-8')
                health_data = json.loads(body)
                
                assert health_data["status"] == "ok"
                assert "timestamp" in health_data
                assert isinstance(health_data["timestamp"], (int, float))
                
                conn.close()
            finally:
                server.shutdown()
                server_thread.join(timeout=1)
    
    def test_ndjson_integrity(self):
        """Test NDJSON log format integrity - no embedded newlines, valid JSON per line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler_class = create_handler(token="test", outdir=tmpdir, rotate_min=0)
            
            server = HTTPServer(("127.0.0.1", 0), handler_class)
            server_thread = Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            try:
                _, port = server.server_address
                
                # Send a test POST request
                conn = HTTPConnection("127.0.0.1", port)
                test_data = {
                    "map": {"game_time": 123},
                    "player": {"gold": 500},
                    "hero": {"health_percent": 0.75}
                }
                
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test"
                }
                
                conn.request("POST", "/", json.dumps(test_data), headers)
                response = conn.getresponse()
                assert response.status == 200
                conn.close()
                
                # Find the log file
                log_files = list(Path(tmpdir).glob("session-*.ndjson"))
                assert len(log_files) == 1
                
                log_file = log_files[0]
                with open(log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check that content ends with newline
                assert content.endswith('\n'), "NDJSON file should end with newline"
                
                # Split into lines (excluding final empty line from trailing newline)
                lines = content.strip().split('\n')
                assert len(lines) >= 2, "Should have session_start + data entry"
                
                # Check each line is valid JSON with no embedded newlines
                for i, line in enumerate(lines):
                    assert '\n' not in line, f"Line {i} contains embedded newline: {line!r}"
                    
                    # Parse as JSON to ensure validity
                    try:
                        data = json.loads(line)
                        assert isinstance(data, dict), f"Line {i} should decode to dict: {line!r}"
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Line {i} is not valid JSON: {line!r} - {e}")
                
                # Check session_start record
                session_start = json.loads(lines[0])
                assert session_start["type"] == "session_start"
                assert "ts" in session_start
                assert "timestamp" in session_start
                
                # Check data record
                data_entry = json.loads(lines[1])
                assert "ts" in data_entry
                assert "data" in data_entry
                assert data_entry["data"]["map"]["game_time"] == 123
                
            finally:
                server.shutdown() 
                server_thread.join(timeout=1)
    
    def test_port_zero_binding(self):
        """Test that port=0 results in automatic port assignment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            handler_class = create_handler(token=None, outdir=tmpdir, rotate_min=0)
            
            # Use port 0 for automatic assignment
            server = HTTPServer(("127.0.0.1", 0), handler_class)
            
            # Check that we got an actual port assigned
            _, actual_port = server.server_address
            assert actual_port > 0, "Port should be assigned automatically"
            assert actual_port != 0, "Port should not remain 0"
            
            server.server_close()
    
    def test_rotation_creates_new_file(self):
        """Test that log rotation creates a new session file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very short rotation for testing (0.01 minutes = 0.6 seconds)
            handler_class = create_handler(token=None, outdir=tmpdir, rotate_min=0.01)
            
            server = HTTPServer(("127.0.0.1", 0), handler_class)
            server_thread = Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            try:
                _, port = server.server_address
                conn = HTTPConnection("127.0.0.1", port)
                
                # Send first request
                test_data = {"map": {"game_time": 100}}
                headers = {"Content-Type": "application/json"}
                conn.request("POST", "/", json.dumps(test_data), headers)
                response = conn.getresponse()
                response.read()  # Consume response
                
                # Check one file exists
                log_files = list(Path(tmpdir).glob("session-*.ndjson"))
                assert len(log_files) == 1
                first_file = log_files[0]
                
                # Wait for rotation time to pass
                time.sleep(0.7)  # Slightly more than 0.6 seconds
                
                # Send second request (should trigger rotation)
                conn.request("POST", "/", json.dumps(test_data), headers)
                response = conn.getresponse()
                response.read()  # Consume response
                
                conn.close()
                
                # Check that we now have two files
                log_files = list(Path(tmpdir).glob("session-*.ndjson"))
                assert len(log_files) == 2, f"Expected 2 log files, got {len(log_files)}"
                
                # Verify both files have content
                for log_file in log_files:
                    content = log_file.read_text()
                    lines = content.strip().split('\n')
                    assert len(lines) >= 1, f"Log file {log_file.name} should have content"
                
            finally:
                server.shutdown()
                server_thread.join(timeout=1)