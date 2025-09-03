#!/usr/bin/env python3
"""
CLI script to run the Dota 2 Game State Integration server.

Usage:
    python -m scripts.gsi_run --port 53000 --token secret --outdir artifacts/gsi
"""

import argparse
import sys
from http.server import HTTPServer
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gsi.server import create_handler


def main():
    """Main entry point for GSI server."""
    parser = argparse.ArgumentParser(
        description="Run Dota 2 Game State Integration server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host address to bind to"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=53000,
        help="Port to listen on"
    )
    
    parser.add_argument(
        "--token", 
        default=None,
        help="Authentication token (optional). If provided, requests must include 'Authorization: Bearer <token>' header or {'auth':{'token':'<token>'}} in JSON body."
    )
    
    parser.add_argument(
        "--outdir", 
        default="artifacts/gsi",
        help="Output directory for NDJSON logs"
    )
    
    parser.add_argument(
        "--rotate-min", 
        type=float,
        default=0,
        help="Rotate log files every N minutes (0 = no rotation)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Create handler class with bound parameters
    handler_class = create_handler(token=args.token, outdir=str(outdir), rotate_min=args.rotate_min)
    
    # Start HTTP server
    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, handler_class)
    
    # Get actual bound port (useful when port=0 for auto-assignment)
    actual_host, actual_port = httpd.server_address
    actual_url = f"http://{actual_host}:{actual_port}/"
    
    print(f"🚀 GSI server starting on {actual_host}:{actual_port}")
    print(f"🌐 Server URL: {actual_url}")
    if args.token:
        print(f"🔒 Authentication required (token: {args.token[:4]}...)")
    else:
        print("⚠️  No authentication - all requests accepted")
    
    if args.rotate_min > 0:
        print(f"🔄 Log rotation: every {args.rotate_min:.1f} minutes")
    
    print(f"📁 Logging to: {outdir.resolve()}")
    print("📡 Waiting for Dota 2 GSI requests...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server shutting down...")
        httpd.shutdown()
        print("✅ Server stopped")


if __name__ == "__main__":
    main()