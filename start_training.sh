#!/bin/bash

# Script to start Pensieve training in the background

cd /mnt/c/Users/91630/OneDrive/Desktop/freelancer/pensieve/sim

echo "Starting Pensieve training..."
echo "This will run in the background and may take several hours."
echo ""

# Start training in background with nohup
nohup python3 multi_agent.py > training.log 2>&1 &

# Get the process ID
PID=$!

echo "Training started with PID: $PID"
echo "Log file: sim/training.log"
echo ""
echo "To monitor progress:"
echo "  tail -f sim/training.log"
echo ""
echo "To check if training is running:"
echo "  ps aux | grep multi_agent"
echo ""
echo "To stop training:"
echo "  kill $PID"
echo ""
echo "Models will be saved to: sim/results/"
echo "Checkpoints saved every 100 epochs"
