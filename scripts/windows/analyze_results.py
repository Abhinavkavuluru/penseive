#!/usr/bin/env python3
"""
Pensieve - Results Analysis Script
Compares performance of BB, MPC, and DP models
"""

import os
import sys

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy not installed")
    print("Please run: pip install numpy")
    sys.exit(1)


def analyze_log(filepath):
    """Parse standard log file (BB, MPC) and compute metrics"""
    data = {
        'timestamps': [],
        'bitrates': [],
        'buffer_sizes': [],
        'rebufs': [],
        'chunk_sizes': [],
        'delays': [],
        'rewards': []
    }

    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 7:
                try:
                    data['timestamps'].append(float(parts[0]))
                    data['bitrates'].append(float(parts[1]))
                    data['buffer_sizes'].append(float(parts[2]))
                    data['rebufs'].append(float(parts[3]))
                    data['chunk_sizes'].append(float(parts[4]))
                    data['delays'].append(float(parts[5]))
                    data['rewards'].append(float(parts[6]))
                except ValueError:
                    continue

    if not data['rewards']:
        return None

    # Calculate metrics
    total_reward = sum(data['rewards'])
    avg_bitrate = np.mean(data['bitrates'])
    total_rebuf = sum(data['rebufs'])
    avg_buffer = np.mean(data['buffer_sizes'])

    # Count quality switches
    num_switches = sum(1 for i in range(1, len(data['bitrates']))
                       if data['bitrates'][i] != data['bitrates'][i-1])

    return {
        'total_reward': total_reward,
        'avg_reward': np.mean(data['rewards']),
        'avg_bitrate': avg_bitrate,
        'total_rebuf': total_rebuf,
        'avg_buffer': avg_buffer,
        'num_switches': num_switches,
        'num_chunks': len(data['rewards'])
    }


def analyze_dp(filepath):
    """Parse DP output (different format - first line is total reward)"""
    if not os.path.exists(filepath):
        return None

    if os.path.getsize(filepath) == 0:
        return None

    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            return {'total_reward': float(first_line)}
    except (ValueError, IOError):
        return None


def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)


def print_model_results(model_name, metrics, is_dp=False):
    """Print results for a single model"""
    print(f"\n{model_name}:")
    print(f"  Total Reward:      {metrics['total_reward']:>8.2f}")

    if not is_dp:
        print(f"  Avg Reward/chunk:  {metrics['avg_reward']:>8.3f}")
        print(f"  Avg Bitrate:       {metrics['avg_bitrate']:>8.0f} Kbps")
        print(f"  Total Rebuffering: {metrics['total_rebuf']:>8.2f} sec")
        print(f"  Avg Buffer Size:   {metrics['avg_buffer']:>8.2f} sec")
        print(f"  Quality Switches:  {metrics['num_switches']:>8}")
        print(f"  Chunks Played:     {metrics['num_chunks']:>8}")


def main():
    """Main analysis function"""
    # Change to test directory
    if os.path.exists('results'):
        os.chdir('.')
    elif os.path.exists('test/results'):
        os.chdir('test')
    else:
        print("ERROR: Cannot find results directory")
        return 1

    traces = ['trace_10mbps', 'trace_5mbps', 'trace_variable']
    schemes = ['bb', 'mpc']

    print_separator()
    print("PENSIEVE - ABR ALGORITHM PERFORMANCE COMPARISON")
    print_separator()
    print()
    print("Analyzing results from test/results/ directory...")
    print()

    # Store all results for comparison
    all_results = {}

    for trace in traces:
        print_separator()
        print(f"Network Trace: {trace.upper()}")
        print_separator()

        trace_results = {}

        # Analyze BB and MPC
        for scheme in schemes:
            filepath = f'results/log_sim_{scheme}_{trace}'
            metrics = analyze_log(filepath)
            if metrics:
                trace_results[scheme.upper()] = metrics
                print_model_results(scheme.upper(), metrics)

        # Analyze DP
        dp_file = f'results/log_sim_dp_{trace}'
        dp_metrics = analyze_dp(dp_file)
        if dp_metrics:
            trace_results['DP (Optimal)'] = dp_metrics
            print_model_results('DP (Optimal)', dp_metrics, is_dp=True)

        all_results[trace] = trace_results

    # Summary comparison
    print()
    print_separator()
    print("SUMMARY - MODEL COMPARISON")
    print_separator()
    print()

    # Compare total rewards across traces
    for trace in traces:
        if trace in all_results and all_results[trace]:
            print(f"\n{trace.upper()}:")
            print("-" * 50)

            results = all_results[trace]
            sorted_results = sorted(results.items(),
                                    key=lambda x: x[1]['total_reward'],
                                    reverse=True)

            for rank, (model, metrics) in enumerate(sorted_results, 1):
                reward = metrics['total_reward']
                if rank == 1:
                    print(f"  {rank}. {model:15s} {reward:8.2f}  ⭐ BEST")
                else:
                    gap = sorted_results[0][1]['total_reward'] - reward
                    print(f"  {rank}. {model:15s} {reward:8.2f}  (-{gap:.2f})")

    # Explanation
    print()
    print_separator()
    print("UNDERSTANDING THE RESULTS")
    print_separator()
    print("""
METRICS EXPLAINED:
------------------
• Total Reward = Quality - 4.3×Rebuffering - Smoothness_Penalty
  → Higher is better (combines quality, rebuffering, smoothness)

• Average Bitrate = Video quality level (300-4300 Kbps)
  → Higher = better video quality

• Total Rebuffering = Time spent buffering (seconds)
  → Lower is better (0 is ideal)

• Quality Switches = Number of bitrate changes
  → Lower = smoother playback experience

• Average Buffer = Video buffer level (seconds)
  → Higher = more stable playback


ALGORITHM TYPES:
----------------
• BB (Buffer-Based):
  - Simple, reactive algorithm
  - Uses buffer thresholds (5s, 15s)
  - Fast execution
  - Good baseline performance

• MPC (Model Predictive Control):
  - Predictive optimization
  - Looks ahead 5 chunks
  - Balances quality vs buffering
  - Usually better than BB

• DP (Dynamic Programming - OFFLINE OPTIMAL):
  - Theoretical upper bound
  - Assumes perfect future knowledge
  - NOT achievable in real-time
  - Used as benchmark target


PERFORMANCE GAPS:
-----------------
The gap between DP and practical algorithms (BB/MPC) represents:
  → Room for improvement with better algorithms
  → Value of prediction and optimization
  → Maximum possible gain from reinforcement learning

If RL model worked, it should:
  → Beat BB and MPC consistently
  → Approach DP performance (but never exceed it)
  → Adapt to different network conditions


WHAT TO LOOK FOR:
-----------------
✓ DP should have highest reward (it has perfect knowledge)
✓ MPC often beats BB (optimization helps)
✓ Lower rebuffering is critical (users hate buffering)
✓ Fewer switches = smoother experience
✓ Different traces favor different strategies
""")

    print_separator()
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
