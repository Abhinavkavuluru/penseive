#!/bin/bash

echo "=========================================="
echo "Pensieve Environment Verification"
echo "=========================================="
echo ""

# Check Python
echo "1. Python Version:"
python3 --version
echo ""

# Check pip packages
echo "2. Installed Python Packages:"
pip3 list | grep -E "(tensorflow|tflearn|numpy|scipy|matplotlib|h5py|selenium)"
echo ""

# Check system tools
echo "3. System Tools:"
echo -n "  Mahimahi: "
which mm-delay > /dev/null 2>&1 && echo "✓ Installed" || echo "✗ Not found"
echo -n "  Apache: "
which apache2 > /dev/null 2>&1 && echo "✓ Installed" || echo "✗ Not found"
echo -n "  Node.js: "
node --version 2>/dev/null || echo "✗ Not found"
echo -n "  npm: "
npm --version 2>/dev/null || echo "✗ Not found"
echo -n "  Chrome: "
google-chrome --version 2>/dev/null || echo "✗ Not found"
echo ""

# Check Apache status
echo "4. Apache Server Status:"
sudo service apache2 status | head -3
echo ""

# Check project directories
echo "5. Project Directories:"
for dir in cooked_traces sim/cooked_traces sim/cooked_test_traces test/cooked_test_traces rl_server/results run_exp/results real_exp/results sim/results test/results test/models; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir" 2>/dev/null | wc -l)
        echo "  ✓ $dir ($count files)"
    else
        echo "  ✗ $dir (missing)"
    fi
done
echo ""

# Check trace files
echo "6. Network Trace Files:"
trace_count=$(find cooked_traces sim/cooked_traces sim/cooked_test_traces test/cooked_test_traces -type f 2>/dev/null | wc -l)
if [ $trace_count -gt 0 ]; then
    echo "  ✓ Found $trace_count trace files"
else
    echo "  ✗ No trace files found - run: bash generate_sample_traces.sh"
fi
echo ""

# Test Python imports
echo "7. Testing Python Imports:"
python3 << EOF
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"  ✗ TensorFlow: {e}")

try:
    import tflearn
    print("  ✓ TFLearn")
except Exception as e:
    print(f"  ✗ TFLearn: {e}")

try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except Exception as e:
    print(f"  ✗ NumPy: {e}")

try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"  ✗ Matplotlib: {e}")

try:
    import scipy
    print(f"  ✓ SciPy {scipy.__version__}")
except Exception as e:
    print(f"  ✗ SciPy: {e}")

try:
    import h5py
    print(f"  ✓ h5py {h5py.__version__}")
except Exception as e:
    print(f"  ✗ h5py: {e}")

try:
    import selenium
    print(f"  ✓ Selenium {selenium.__version__}")
except Exception as e:
    print(f"  ✗ Selenium: {e}")
EOF

echo ""
echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
