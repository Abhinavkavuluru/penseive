#!/bin/bash

#===============================================================================
# QUICK TRAINING SCRIPT - Run A3C Training Only
# Usage: bash run_training.sh
#===============================================================================

echo "========================================"
echo "  Starting A3C Training"
echo "========================================"

cd "$(dirname "$0")/sim"

# Check if traces exist
if [ ! -d "cooked_traces" ] || [ -z "$(ls -A cooked_traces 2>/dev/null)" ]; then
    echo "ERROR: No training traces found!"
    echo "Run setup_and_run.sh first to create traces."
    exit 1
fi

echo "Training traces found: $(ls cooked_traces | wc -l) files"
echo ""
echo "Starting training..."
echo "Logs: sim/results/log_central"
echo "Press Ctrl+C to stop"
echo ""

python3 multi_agent.py
