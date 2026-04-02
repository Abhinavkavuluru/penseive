#!/usr/bin/env python3
"""
Cross-platform launcher for Pensieve video streaming experiment.
Starts both the video server and the RL ABR server, then opens the browser.

Works on macOS, Linux, and Windows.

Usage:
    python run_streaming.py                  # defaults: RL algo, port 8080
    python run_streaming.py --algo RL        # use RL-based ABR
    python run_streaming.py --algo fastMPC   # use fast MPC
    python run_streaming.py --algo robustMPC # use robust MPC
    python run_streaming.py --algo BOLA      # use BOLA
    python run_streaming.py --port 9000      # custom video server port
    python run_streaming.py --no-browser     # don't auto-open browser

Requirements:
    pip install tensorflow numpy selenium webdriver-manager
"""
import argparse
import http.server
import os
import platform
import signal
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from time import sleep


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_SERVER_DIR = os.path.join(SCRIPT_DIR, 'video_server')
RL_SERVER_DIR = os.path.join(SCRIPT_DIR, 'rl_server')


def start_video_server(port, directory):
    """Start a simple HTTP server to serve video chunks."""
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=directory)

    class QuietHandler(handler):
        def log_message(self, format, *args):
            pass  # suppress request logs

    server = http.server.HTTPServer(('0.0.0.0', port), QuietHandler)
    print(f"  Video server: http://localhost:{port}/")
    server.serve_forever()


def start_abr_server(algo):
    """Start the ABR algorithm server on port 8333."""
    python_cmd = sys.executable

    if algo == 'RL':
        script = os.path.join(RL_SERVER_DIR, 'rl_server_no_training.py')
        cmd = [python_cmd, script]
    elif algo == 'fastMPC':
        script = os.path.join(RL_SERVER_DIR, 'mpc_server.py')
        cmd = [python_cmd, script]
    elif algo == 'robustMPC':
        script = os.path.join(RL_SERVER_DIR, 'robust_mpc_server.py')
        cmd = [python_cmd, script]
    else:
        script = os.path.join(RL_SERVER_DIR, 'simple_server.py')
        cmd = [python_cmd, script, algo]

    proc = subprocess.Popen(cmd, cwd=RL_SERVER_DIR)
    return proc


def update_html_port(video_server_port):
    """The HTML pages reference localhost:8333 for ABR server (hardcoded in dash.js).
    The video content is served from the video server port.
    We need to ensure the HTML page is loaded from the video server."""
    pass  # The HTML is served from video_server_dir, ABR port 8333 is in dash.js


def main():
    parser = argparse.ArgumentParser(description='Pensieve Video Streaming Experiment')
    parser.add_argument('--algo', type=str, default='RL',
                        choices=['RL', 'fastMPC', 'robustMPC', 'BOLA', 'BB', 'RB', 'FESTIVE'],
                        help='ABR algorithm to use (default: RL)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Video server port (default: 8080)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not auto-open browser')
    args = parser.parse_args()

    print("=" * 60)
    print("  Pensieve Video Streaming Experiment")
    print("=" * 60)
    print(f"  ABR Algorithm: {args.algo}")
    print(f"  Platform:      {platform.system()} {platform.machine()}")
    print()

    # Start video server in background thread
    print("[1/3] Starting video server...")
    video_thread = threading.Thread(
        target=start_video_server,
        args=(args.port, VIDEO_SERVER_DIR),
        daemon=True
    )
    video_thread.start()
    sleep(1)

    # Start ABR server
    print(f"[2/3] Starting {args.algo} ABR server on port 8333...")
    abr_proc = start_abr_server(args.algo)
    sleep(5 if args.algo == 'RL' else 2)  # RL needs time to load TF model

    # Open browser
    url = f'http://localhost:{args.port}/myindex_{args.algo}.html'
    print(f"[3/3] Streaming URL: {url}")

    if not args.no_browser:
        print("  Opening browser...")
        webbrowser.open(url)

    print()
    print("=" * 60)
    print("  Streaming is running! Press Ctrl+C to stop.")
    print("=" * 60)

    # Wait for Ctrl+C
    try:
        abr_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        abr_proc.terminate()
        try:
            abr_proc.wait(timeout=5)
        except Exception:
            abr_proc.kill()
        print("Done.")


if __name__ == '__main__':
    main()
