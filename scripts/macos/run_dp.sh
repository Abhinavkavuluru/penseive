#!/bin/bash
# Pensieve - Run Dynamic Programming (DP) Model
# Computes offline-optimal solution with perfect future knowledge

echo "================================================================================"
echo "Pensieve - Dynamic Programming (DP) - Offline Optimal"
echo "================================================================================"
echo ""
echo "ABOUT THIS MODEL:"
echo "  - Computes the THEORETICAL OPTIMAL solution"
echo "  - Uses dynamic programming algorithm"
echo "  - Assumes PERFECT knowledge of future bandwidth"
echo "  - This is the upper bound - not achievable in real-time"
echo "  - Used as benchmark to evaluate other algorithms"
echo ""
echo "EXECUTION TIME: ~5-10 minutes (much slower than other models)"
echo ""
echo "NOTE: Requires C++ compiler (g++ from Xcode Command Line Tools)"
echo ""
echo "================================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../test" || exit 1

# Check if we're in the right directory
if [ ! -f dp.cc ]; then
    echo "ERROR: Cannot find dp.cc"
    echo "Make sure you're running this from the pensieve repository"
    exit 1
fi

echo "Step 1: Compiling C++ code..."
echo ""

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ compiler not found!"
    echo ""
    echo "Please install Xcode Command Line Tools:"
    echo "  xcode-select --install"
    echo ""
    exit 1
fi

echo "Using g++ compiler:"
g++ --version | head -1

# Compile
g++ -std=c++11 -O3 dp.cc -o dp
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "Compilation successful!"
echo ""

echo "================================================================================"
echo "Step 2: Running DP algorithm..."
echo "================================================================================"
echo ""
echo "Network traces being tested:"
echo "  - trace_10mbps (high bandwidth)"
echo "  - trace_5mbps (medium bandwidth)"
echo "  - trace_variable (varying bandwidth)"
echo ""
echo "This will take several minutes. Progress:"
echo ""

# Run the model
./dp

echo ""
echo "================================================================================"
echo "DP Model Completed!"
echo "================================================================================"
echo ""
echo "Results saved to: test/results/"
echo "  - log_sim_dp_trace_10mbps"
echo "  - log_sim_dp_trace_5mbps"
echo "  - log_sim_dp_trace_variable"
echo ""
echo "DP output format (different from other models):"
echo "  First line: OPTIMAL total reward"
echo "  Following lines: Optimal decision sequence (reverse order)"
echo ""
echo "This is the THEORETICAL UPPER BOUND:"
echo "  - Shows best possible performance with perfect knowledge"
echo "  - Real-time algorithms (BB, MPC, RL) should approach this"
echo "  - Gap between DP and real-time = room for improvement"
echo ""
echo "Next: Run ./analyze_results.sh to see how BB and MPC compare to optimal"
echo ""
