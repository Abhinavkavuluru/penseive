#!/bin/bash

# Script to start Pensieve RL model training

cd /mnt/c/Users/91630/OneDrive/Desktop/freelancer/pensieve/sim

echo "=========================================="
echo "Starting Pensieve RL Model Training"
echo "=========================================="
echo ""
echo "Training Configuration:"
echo "  - Algorithm: A3C (Asynchronous Advantage Actor-Critic)"
echo "  - Agents: 16 parallel workers"
echo "  - Training sequence length: 100 steps"
echo "  - Model save interval: Every 100 epochs"
echo "  - Network traces: 3 files in cooked_traces/"
echo ""
echo "This will take several hours. Training will run in background."
echo ""

# Start training in background
nohup python3 multi_agent.py > training.log 2>&1 &

# Get PID
PID=$!

echo "Training started with PID: $PID"
echo "Log file: sim/training.log"
echo ""
echo "Monitor progress:"
echo "  tail -f sim/training.log"
echo ""
echo "Check if running:"
echo "  ps aux | grep multi_agent"
echo ""
echo "Stop training:"
echo "  kill $PID"
echo ""
echo "Models will be saved to: sim/results/"
echo "Checkpoints saved every 100 epochs"
echo ""
echo "Expected training time: Several hours to days"
echo "Recommendation: Let it run for at least 1000-5000 epochs"
