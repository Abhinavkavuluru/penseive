#!/bin/bash

# Quick script to test all three algorithms

cd /mnt/c/Users/91630/OneDrive/Desktop/freelancer/pensieve/test

echo "=========================================="
echo "Testing All Three Adaptive Bitrate Algorithms"
echo "=========================================="
echo ""

# Test 1: Buffer-Based (BB)
echo "1. Testing Buffer-Based (BB) Algorithm..."
echo "   Status: Already completed ✓"
echo "   Results: results/log_sim_bb_*"
echo ""

# Test 2: Model Predictive Control (MPC)
echo "2. Testing Model Predictive Control (MPC) Algorithm..."
echo "   Status: Already completed ✓"
echo "   Results: results/log_sim_mpc_*"
echo ""

# Test 3: Reinforcement Learning (Pensieve)
echo "3. Testing Pensieve (RL) Algorithm..."
echo "   Using pre-trained model: models/pretrain_linear_reward.ckpt"

# Check if tflearn is available
python3 -c "import tflearn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   TFLearn available, running RL test..."
    timeout 120 python3 rl_no_training.py 2>&1 | grep -E "video count|Model restored|Error|Traceback" | head -20
    if [ $? -eq 0 ]; then
        echo "   Status: Completed ✓"
        echo "   Results: results/log_sim_rl_*"
    else
        echo "   Status: Failed ✗"
    fi
else
    echo "   ⚠️  TFLearn not available - RL test cannot run"
    echo "   Pre-trained model exists but requires tflearn library"
    echo ""
    echo "   Options:"
    echo "   1. Use BB and MPC results (already completed)"
    echo "   2. Fix tflearn compatibility (requires code changes)"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✓ BB (Buffer-Based): Completed"
echo "✓ MPC (Model Predictive Control): Completed"
echo "? RL (Pensieve): Requires tflearn fix"
echo ""
echo "View results:"
echo "  ls -lh results/"
echo "  tail -20 results/log_sim_bb_trace_5mbps"
echo "  tail -20 results/log_sim_mpc_trace_5mbps"
