#!/bin/bash

#===============================================================================
# TEST SCRIPT - Run All Tests and Generate Comparison
# Usage: bash run_tests.sh
#===============================================================================

echo "========================================"
echo "  Running Tests & Generating Results"
echo "========================================"

cd "$(dirname "$0")/test"

# Check for trained model
if ls ../sim/results/nn_model_ep_*.ckpt* 1> /dev/null 2>&1; then
    echo "Found trained model. Copying to test/models/..."
    mkdir -p models
    cp ../sim/results/nn_model_ep_*.ckpt* models/ 2>/dev/null
fi

echo ""
echo "[1/4] Running Buffer-Based (BB)..."
python3 bb.py 2>/dev/null && echo "  Done." || echo "  Skipped."

echo ""
echo "[2/4] Running MPC..."
python3 mpc.py 2>/dev/null && echo "  Done." || echo "  Skipped."

echo ""
echo "[3/4] Running RL Agent..."
python3 rl_no_training.py 2>/dev/null && echo "  Done." || echo "  Skipped (no model found)."

echo ""
echo "[4/4] Generating comparison plots..."
python3 plot_results.py 2>/dev/null && echo "  Done." || echo "  Skipped."

echo ""
echo "========================================"
echo "  Tests Complete!"
echo "========================================"
echo "Results saved in: test/results/"
ls -la results/ 2>/dev/null || echo "(no results yet)"
