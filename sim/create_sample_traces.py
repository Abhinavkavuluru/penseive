"""
Create sample network traces for training ABR algorithms
These simulate various network conditions: LTE, 3G, WiFi, etc.
"""

import os
import numpy as np

TRACE_DIR = './cooked_traces/'

def create_trace(filename, duration_sec=300, pattern='lte'):
    """
    Create a network trace file

    Format: time_stamp (seconds) throughput (Mbps)
    """
    np.random.seed(hash(filename) % 2**32)

    times = []
    bandwidths = []

    t = 0
    dt = 1.0  # 1 second intervals

    while t < duration_sec:
        if pattern == 'lte':
            # LTE: 5-20 Mbps with moderate variation
            bw = np.random.uniform(5, 20) + np.sin(t/30) * 3
        elif pattern == '3g':
            # 3G: 0.5-3 Mbps with high variation
            bw = np.random.uniform(0.5, 3) + np.random.randn() * 0.5
        elif pattern == 'wifi':
            # WiFi: 10-50 Mbps, occasional drops
            bw = np.random.uniform(10, 50)
            if np.random.random() < 0.1:  # 10% chance of drop
                bw *= 0.2
        elif pattern == 'cable':
            # Cable: 5-15 Mbps, stable
            bw = np.random.uniform(5, 15) + np.random.randn() * 1
        elif pattern == 'variable':
            # Highly variable: 1-30 Mbps
            bw = np.random.uniform(1, 30) + np.sin(t/10) * 5
        else:
            bw = np.random.uniform(2, 10)

        bw = max(0.1, bw)  # Minimum 0.1 Mbps

        times.append(t)
        bandwidths.append(bw)

        t += dt

    # Write trace file
    filepath = os.path.join(TRACE_DIR, filename)
    with open(filepath, 'w') as f:
        for t, bw in zip(times, bandwidths):
            f.write(f"{t}\t{bw}\n")

    print(f"Created: {filename} ({pattern}, {len(times)} samples)")
    return filepath


def main():
    # Create trace directory
    os.makedirs(TRACE_DIR, exist_ok=True)

    # Create diverse set of traces
    traces = [
        # LTE traces
        ('trace_lte_1', 'lte'),
        ('trace_lte_2', 'lte'),
        ('trace_lte_3', 'lte'),
        # 3G traces
        ('trace_3g_1', '3g'),
        ('trace_3g_2', '3g'),
        # WiFi traces
        ('trace_wifi_1', 'wifi'),
        ('trace_wifi_2', 'wifi'),
        # Cable traces
        ('trace_cable_1', 'cable'),
        ('trace_cable_2', 'cable'),
        # Variable traces
        ('trace_variable_1', 'variable'),
        ('trace_variable_2', 'variable'),
        ('trace_variable_3', 'variable'),
    ]

    for filename, pattern in traces:
        create_trace(filename, duration_sec=300, pattern=pattern)

    print(f"\nCreated {len(traces)} trace files in {TRACE_DIR}")


if __name__ == '__main__':
    main()
