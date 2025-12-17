#!/bin/bash
# Pensieve - Run All Models
# Executes BB, MPC, and DP models sequentially

set -e  # Exit on error

echo "================================================================================"
echo "Pensieve - Run All ABR Models"
echo "================================================================================"
echo ""
echo "This script will run all three models:"
echo "  1. Buffer-Based (BB) - ~1-2 minutes"
echo "  2. Model Predictive Control (MPC) - ~2-3 minutes"
echo "  3. Dynamic Programming (DP) - ~5-10 minutes"
echo ""
echo "Total estimated time: 10-15 minutes"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Record start time
START_TIME=$(date +%s)
echo "Started at: $(date)"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../test" || exit 1

if [ ! -f bb.py ]; then
    echo "ERROR: Cannot find test directory"
    exit 1
fi

echo "================================================================================"
echo "[1/3] Running Buffer-Based (BB) Model"
echo "================================================================================"
echo ""

python3 bb.py
if [ $? -ne 0 ]; then
    echo "ERROR: BB model failed"
    exit 1
fi

echo ""
echo "BB completed successfully!"
echo ""
sleep 2

echo "================================================================================"
echo "[2/3] Running Model Predictive Control (MPC)"
echo "================================================================================"
echo ""

python3 mpc.py
if [ $? -ne 0 ]; then
    echo "ERROR: MPC model failed"
    exit 1
fi

echo ""
echo "MPC completed successfully!"
echo ""
sleep 2

echo "================================================================================"
echo "[3/3] Running Dynamic Programming (DP) - Offline Optimal"
echo "================================================================================"
echo ""

# Check for C++ compiler
if ! command -v g++ &> /dev/null; then
    echo "WARNING: g++ not found. Skipping DP model."
    echo "Install with: xcode-select --install"
else
    echo "Compiling DP..."
    g++ -std=c++11 -O3 dp.cc -o dp

    if [ $? -eq 0 ]; then
        echo "Running DP..."
        ./dp
        if [ $? -ne 0 ]; then
            echo "WARNING: DP model failed"
        else
            echo ""
            echo "DP completed successfully!"
        fi
    else
        echo "WARNING: DP compilation failed. Skipping DP model."
    fi
fi

echo ""
echo "================================================================================"
echo "All Models Completed!"
echo "================================================================================"
echo ""

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "Finished at: $(date)"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results location: test/results/"
echo ""
ls -1 results/log_sim_* 2>/dev/null || echo "No results found"
echo ""

echo "================================================================================"
echo "Next Steps:"
echo "================================================================================"
echo ""
echo "Run ./analyze_results.sh to see detailed performance comparison"
echo ""

# Ask if user wants to analyze now
read -p "Would you like to analyze results now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$SCRIPT_DIR"
    ./analyze_results.sh
fi
