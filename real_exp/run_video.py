#!/usr/bin/env python3
"""
Cross-platform browser-based video streaming experiment using Selenium.
Works on macOS, Linux, and Windows.

Requirements:
    pip install selenium webdriver-manager

Usage:
    python run_video.py <abr_algo> <run_time_sec> <exp_id>
    Example: python run_video.py RL 280 0
"""
import http.server
import os
import sys
import subprocess
import platform
import shutil
import tempfile
import threading
from functools import partial
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# Try webdriver-manager for automatic chromedriver management (cross-platform)
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False

# Video server port (must not conflict with ABR server on 8333)
VIDEO_SERVER_PORT = 8333 + 1  # 8334


def start_video_server():
    """Start HTTP server to serve video chunks from video_server/ directory."""
    video_dir = os.path.join(os.path.dirname(__file__), '..', 'video_server')
    video_dir = os.path.abspath(video_dir)

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=video_dir, **kwargs)

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(('0.0.0.0', VIDEO_SERVER_PORT), QuietHandler)
    server.serve_forever()


def kill_process(proc):
    """Kill a subprocess cross-platform."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    abr_algo = sys.argv[1]
    run_time = int(sys.argv[2])
    exp_id = sys.argv[3]

    proc = None
    driver = None
    video_thread = None

    try:
        # Start video server in background thread
        video_thread = threading.Thread(target=start_video_server, daemon=True)
        video_thread.start()
        sleep(1)

        url = f'http://localhost:{VIDEO_SERVER_PORT}/myindex_{abr_algo}.html'

        # copy over the chrome user dir (cross-platform)
        default_chrome_user_dir = os.path.join(os.path.dirname(__file__), '..', 'abr_browser_dir', 'chrome_data_dir')
        chrome_user_dir = os.path.join(tempfile.gettempdir(), 'chrome_user_dir_real_exp_' + abr_algo)
        if os.path.exists(chrome_user_dir):
            shutil.rmtree(chrome_user_dir, ignore_errors=True)
        if os.path.exists(default_chrome_user_dir):
            shutil.copytree(default_chrome_user_dir, chrome_user_dir)
        else:
            os.makedirs(chrome_user_dir, exist_ok=True)

        # start abr algorithm server
        python_cmd = sys.executable
        rl_server_dir = os.path.join(os.path.dirname(__file__), '..', 'rl_server')

        if abr_algo == 'RL':
            server_script = os.path.join(rl_server_dir, 'rl_server_no_training.py')
            command = [python_cmd, server_script, exp_id]
        elif abr_algo == 'fastMPC':
            server_script = os.path.join(rl_server_dir, 'mpc_server.py')
            command = [python_cmd, server_script, exp_id]
        elif abr_algo == 'robustMPC':
            server_script = os.path.join(rl_server_dir, 'robust_mpc_server.py')
            command = [python_cmd, server_script, exp_id]
        else:
            server_script = os.path.join(rl_server_dir, 'simple_server.py')
            command = [python_cmd, server_script, abr_algo, exp_id]

        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # RL server needs more time to load TF model
        sleep(8 if abr_algo == 'RL' else 2)

        # initialize chrome driver (cross-platform)
        options = Options()
        options.add_argument('--user-data-dir=' + chrome_user_dir)
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--autoplay-policy=no-user-gesture-required')

        # headless mode: use --headless for environments without display
        # Uncomment the next line to run headless (works on all platforms):
        # options.add_argument('--headless=new')

        if USE_WEBDRIVER_MANAGER:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
        else:
            # Fallback: try system chromedriver or bundled one
            bundled_driver = os.path.join(os.path.dirname(__file__), '..', 'abr_browser_dir', 'chromedriver')
            if platform.system() == 'Windows':
                bundled_driver += '.exe'
            if os.path.exists(bundled_driver):
                service = Service(bundled_driver)
                driver = webdriver.Chrome(service=service, options=options)
            else:
                # Let Selenium find chromedriver on PATH
                driver = webdriver.Chrome(options=options)

        # run chrome
        driver.set_page_load_timeout(30)
        driver.get(url)

        sleep(run_time)

        driver.quit()
        driver = None

        # kill abr algorithm server
        kill_process(proc)
        proc = None

        print('done')

    except Exception as e:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass
        try:
            if proc:
                kill_process(proc)
        except Exception:
            pass

        print(e)


if __name__ == '__main__':
    main()
