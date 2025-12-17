#!/bin/bash

# Simple script to create basic network trace files for testing

echo "Creating simple network trace files..."

# Create directories
mkdir -p cooked_traces
mkdir -p sim/cooked_traces
mkdir -p sim/cooked_test_traces
mkdir -p test/cooked_test_traces

# Create a simple Python script to generate traces
cat > /tmp/create_traces.py << 'EOF'
import random

def create_trace(filename, duration=60, avg_bandwidth=5.0, variance=2.0):
    """Create a simple network trace file"""
    with open(filename, 'w') as f:
        for t in range(duration):
            # Generate bandwidth with some randomness
            bandwidth = max(0.5, avg_bandwidth + random.gauss(0, variance))
            f.write(f"{t} {bandwidth:.2f}\n")

# Create several trace files with different characteristics
traces = [
    ("trace_5mbps", 60, 5.0, 1.0),
    ("trace_10mbps", 60, 10.0, 2.0),
    ("trace_variable", 60, 7.0, 3.0),
]

for name, duration, avg_bw, var in traces:
    create_trace(name, duration, avg_bw, var)
    print(f"Created {name}")

print("Done!")
EOF

# Run the Python script
cd cooked_traces
python3 /tmp/create_traces.py

# Copy traces to other directories
echo "Copying traces to other directories..."
cp * ../sim/cooked_traces/
cp * ../sim/cooked_test_traces/
cp * ../test/cooked_test_traces/

cd ..

echo ""
echo "Trace files created:"
echo "  cooked_traces/: $(ls -1 cooked_traces 2>/dev/null | wc -l) files"
echo "  sim/cooked_traces/: $(ls -1 sim/cooked_traces 2>/dev/null | wc -l) files"
echo "  sim/cooked_test_traces/: $(ls -1 sim/cooked_test_traces 2>/dev/null | wc -l) files"
echo "  test/cooked_test_traces/: $(ls -1 test/cooked_test_traces 2>/dev/null | wc -l) files"
echo ""
echo "Sample trace files created successfully!"
