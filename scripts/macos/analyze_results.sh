#!/bin/bash
# Pensieve - Analyze Results
# Compares performance of all ABR models

echo "================================================================================"
echo "Pensieve - Results Analysis"
echo "================================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/../../test" || exit 1

# Check if results exist
if [ ! -f results/log_sim_bb_trace_10mbps ]; then
    echo "ERROR: No results found!"
    echo "Please run the models first:"
    echo "  - ./run_bb.sh"
    echo "  - ./run_mpc.sh"
    echo "  - ./run_dp.sh"
    echo "Or run all at once: ./run_all_models.sh"
    echo ""
    exit 1
fi

echo "Analyzing results..."
echo ""

# Run Python analysis script
python3 "$SCRIPT_DIR/analyze_results.py"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Analysis failed"
    echo "Make sure NumPy is installed: pip3 install numpy"
    exit 1
fi

echo ""
echo "================================================================================"
echo "Analysis Complete!"
echo "================================================================================"
echo ""
echo "Key Takeaways:"
echo "  - Higher Total Reward = Better overall performance"
echo "  - DP shows theoretical upper bound (offline optimal)"
echo "  - Gap between DP and others = room for improvement"
echo "  - Fewer quality switches = smoother playback"
echo "  - Lower rebuffering = better user experience"
echo ""
