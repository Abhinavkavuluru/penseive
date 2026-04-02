#!/usr/bin/env python3
"""
Cross-platform experiment runner for Pensieve ABR algorithms.
Works on macOS, Linux, and Windows.

Usage:
    python run_exp.py
"""
import sys
import os
import subprocess
import numpy as np


RUN_SCRIPT = 'run_video.py'
RANDOM_SEED = 42
RUN_TIME = 280  # sec
ABR_ALGO = ['fastMPC', 'robustMPC', 'BOLA', 'RL']
REPEAT_TIME = 10


def main():

    np.random.seed(RANDOM_SEED)

    with open('./chrome_retry_log', 'w') as log:
        log.write('chrome retry log\n')
        log.flush()

        for rt in range(REPEAT_TIME):
            np.random.shuffle(ABR_ALGO)
            for abr_algo in ABR_ALGO:

                while True:

                    script = [sys.executable, RUN_SCRIPT, abr_algo, str(RUN_TIME), str(rt)]

                    proc = subprocess.Popen(script,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

                    (out, err) = proc.communicate()
                    out = out.decode('utf-8', errors='replace')

                    if out.strip() == 'done':
                        break
                    else:
                        log.write(abr_algo + '_' + str(rt) + '\n')
                        log.write(out + '\n')
                        log.flush()



if __name__ == '__main__':
    main()
