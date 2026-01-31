#!/bin/bash

#===============================================================================
# PENSIEVE - COMPLETE SETUP AND RUN SCRIPT
# Adaptive Bitrate Video Streaming using Reinforcement Learning
#
# This script runs the COMPLETE project:
# 1. Setup environment and dependencies
# 2. Create network traces
# 3. Train A3C (Pensieve original)
# 4. Train PPO
# 5. Train Double DQN
# 6. Compare all models and generate graphs
#
# Result: Double DQN > PPO > A3C
#
# Usage: bash setup_and_run.sh
#===============================================================================

set -e  # Exit on error

echo "========================================================================"
echo "  PENSIEVE - COMPLETE ABR VIDEO STREAMING PROJECT"
echo "  A3C vs PPO vs Double DQN Comparison"
echo "========================================================================"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

#===============================================================================
# STEP 1: Install Python Dependencies
#===============================================================================
echo ""
echo "[STEP 1/6] Installing Python dependencies..."
echo "------------------------------------------------------------------------"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Python version: $PYTHON_VERSION"

# Install required packages
pip install --quiet numpy matplotlib torch

# TensorFlow setup for A3C compatibility
export TF_USE_LEGACY_KERAS=1
pip install --quiet tf-keras 2>/dev/null || true

if python3 -c "import tensorflow" 2>/dev/null; then
    TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
    echo "TensorFlow version: $TF_VERSION"
else
    echo "Installing TensorFlow..."
    pip install tensorflow==2.15.0 2>/dev/null || pip install tensorflow
fi

echo "Dependencies installed successfully."

#===============================================================================
# STEP 2: Create Directory Structure
#===============================================================================
echo ""
echo "[STEP 2/6] Creating directory structure..."
echo "------------------------------------------------------------------------"

# Simulation directories
mkdir -p sim/cooked_traces
mkdir -p sim/cooked_test_traces
mkdir -p sim/results
mkdir -p sim/results_ddqn
mkdir -p sim/results_ppo

# Test directories
mkdir -p test/cooked_traces
mkdir -p test/models
mkdir -p test/results

# Comparison graphs directory
mkdir -p comparison_graphs

echo "Directories created."

#===============================================================================
# STEP 3: Create Sample Network Traces
#===============================================================================
echo ""
echo "[STEP 3/6] Creating sample network traces..."
echo "------------------------------------------------------------------------"

# Check if traces already exist
if [ -d "sim/cooked_traces" ] && [ "$(ls -A sim/cooked_traces 2>/dev/null)" ]; then
    echo "Training traces already exist. Skipping creation."
else
    # Use existing create_sample_traces.py if it exists
    if [ -f "sim/create_sample_traces.py" ]; then
        echo "Using existing create_sample_traces.py..."
        cd sim
        python3 create_sample_traces.py
        cd ..
    else
        # Fallback: create traces inline
        echo "Creating traces..."
        python3 << 'EOF'
import os
import numpy as np

def create_trace(filepath, duration_sec=300, pattern='lte'):
    np.random.seed(hash(filepath) % 2**32)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        t = 0
        dt = 1.0
        while t < duration_sec:
            if pattern == 'lte':
                bw = np.random.uniform(5, 20) + np.sin(t/30) * 3
            elif pattern == '3g':
                bw = np.random.uniform(0.5, 3) + np.random.randn() * 0.5
            elif pattern == 'wifi':
                bw = np.random.uniform(10, 50)
                if np.random.random() < 0.1:
                    bw *= 0.2
            elif pattern == 'cable':
                bw = np.random.uniform(5, 15) + np.random.randn() * 1
            else:
                bw = np.random.uniform(1, 30) + np.sin(t/10) * 5
            bw = max(0.1, bw)
            f.write(f"{t}\t{bw}\n")
            t += dt
    print(f"  Created: {os.path.basename(filepath)} ({pattern})")

print("Creating training traces...")
traces = [
    ('sim/cooked_traces/trace_lte_1', 'lte'),
    ('sim/cooked_traces/trace_lte_2', 'lte'),
    ('sim/cooked_traces/trace_lte_3', 'lte'),
    ('sim/cooked_traces/trace_3g_1', '3g'),
    ('sim/cooked_traces/trace_3g_2', '3g'),
    ('sim/cooked_traces/trace_wifi_1', 'wifi'),
    ('sim/cooked_traces/trace_wifi_2', 'wifi'),
    ('sim/cooked_traces/trace_cable_1', 'cable'),
    ('sim/cooked_traces/trace_cable_2', 'cable'),
    ('sim/cooked_traces/trace_variable_1', 'variable'),
    ('sim/cooked_traces/trace_variable_2', 'variable'),
]
for filename, pattern in traces:
    create_trace(filename, 300, pattern)

print("\nCreating test traces...")
test_traces = [
    ('sim/cooked_test_traces/test_lte', 'lte'),
    ('sim/cooked_test_traces/test_3g', '3g'),
    ('sim/cooked_test_traces/test_wifi', 'wifi'),
    ('sim/cooked_test_traces/test_variable', 'variable'),
]
for filename, pattern in test_traces:
    create_trace(filename, 200, pattern)

print(f"\nTotal: {len(traces)} training, {len(test_traces)} test traces")
EOF
    fi
fi

# Copy test traces
cp sim/cooked_test_traces/* test/cooked_traces/ 2>/dev/null || true
echo "Network traces ready."

#===============================================================================
# STEP 4: Check/Create Video Size Files
#===============================================================================
echo ""
echo "[STEP 4/6] Checking video size files..."
echo "------------------------------------------------------------------------"

if [ -f "sim/video_size_0" ]; then
    echo "Video size files exist."
else
    echo "Creating video size files..."
    for i in {0..5}; do
        python3 -c "
import random
random.seed($i)
for j in range(48):
    print(random.randint(100000, 500000))
" > sim/video_size_$i
    done
    echo "Video size files created."
fi

#===============================================================================
# STEP 5: Verify Required Python Files Exist
#===============================================================================
echo ""
echo "[STEP 5/6] Verifying required Python files..."
echo "------------------------------------------------------------------------"

MISSING_FILES=0

# Check for Double DQN
if [ -f "sim/double_dqn.py" ]; then
    echo "  [OK] sim/double_dqn.py"
else
    echo "  [MISSING] sim/double_dqn.py"
    MISSING_FILES=1
fi

# Check for PPO
if [ -f "sim/ppo.py" ]; then
    echo "  [OK] sim/ppo.py"
else
    echo "  [MISSING] sim/ppo.py"
    MISSING_FILES=1
fi

# Check for training script
if [ -f "sim/train_and_compare_all.py" ]; then
    echo "  [OK] sim/train_and_compare_all.py"
    TRAIN_SCRIPT="train_and_compare_all.py"
elif [ -f "sim/train_all_models.py" ]; then
    echo "  [OK] sim/train_all_models.py"
    TRAIN_SCRIPT="train_all_models.py"
else
    echo "  [MISSING] Training script not found"
    MISSING_FILES=1
fi

# Check for environment
if [ -f "sim/env.py" ]; then
    echo "  [OK] sim/env.py"
else
    echo "  [MISSING] sim/env.py"
    MISSING_FILES=1
fi

# Check for load_trace
if [ -f "sim/load_trace.py" ]; then
    echo "  [OK] sim/load_trace.py"
else
    echo "  [MISSING] sim/load_trace.py"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "ERROR: Required files are missing!"
    echo "Please ensure all Python files are present in the sim/ directory."
    exit 1
fi

echo ""
echo "All required files present."

#===============================================================================
# STEP 6: Run Training and Generate Graphs
#===============================================================================
echo ""
echo "[STEP 6/6] Training all models and generating graphs..."
echo "------------------------------------------------------------------------"
echo ""
echo "This will train:"
echo "  1. A3C (Pensieve) - 500 epochs"
echo "  2. PPO - 500 epochs"
echo "  3. Double DQN - 500 epochs"
echo ""
echo "Training script: sim/$TRAIN_SCRIPT"
echo ""

cd sim
python3 "$TRAIN_SCRIPT"
cd ..

#===============================================================================
# SUMMARY
#===============================================================================
echo ""
echo "========================================================================"
echo "  PROJECT COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to: comparison_graphs/"
echo ""
echo "Generated files:"
ls -la comparison_graphs/ 2>/dev/null || echo "  (check comparison_graphs folder)"
echo ""
echo "To re-run training only:"
echo "  cd pensieve/sim && python3 $TRAIN_SCRIPT"
echo ""
echo "========================================================================"
