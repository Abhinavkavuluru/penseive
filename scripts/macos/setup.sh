#!/bin/bash
# Pensieve - macOS Setup Script
# This script installs all required Python dependencies

set -e  # Exit on error

echo "================================================================================"
echo "Pensieve - Setup Script for macOS"
echo "================================================================================"
echo ""
echo "This script will install the required Python packages."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python using one of these methods:"
    echo "  1. Homebrew: brew install python@3.11"
    echo "  2. Download from: https://www.python.org/downloads/"
    exit 1
fi

echo "Python found:"
python3 --version
echo ""

# Check for g++ (needed for DP model)
if ! command -v g++ &> /dev/null; then
    echo "WARNING: g++ not found. DP model will not work."
    echo "To install: xcode-select --install"
    echo ""
    read -p "Continue without g++? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "C++ compiler found:"
    g++ --version | head -1
    echo ""
fi

echo "Installing required packages..."
echo ""

# Upgrade pip
echo "[1/5] Upgrading pip..."
python3 -m pip install --upgrade pip

# Install NumPy
echo "[2/5] Installing NumPy..."
pip3 install numpy==1.26.4

# Install TensorFlow with correct version
echo "[3/5] Installing TensorFlow 2.14.0..."
pip3 install tensorflow==2.14.0

# Install ml_dtypes (critical for compatibility)
echo "[4/5] Installing ml_dtypes 0.2.0..."
pip3 install ml_dtypes==0.2.0

# Install matplotlib for plotting
echo "[5/5] Installing matplotlib..."
pip3 install matplotlib

echo ""
echo "================================================================================"
echo "Installation Complete!"
echo "================================================================================"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import numpy; print('NumPy version:', numpy.__version__)"
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python3 -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Run individual models: ./run_bb.sh, ./run_mpc.sh, ./run_dp.sh"
echo "  2. Run all models: ./run_all_models.sh"
echo "  3. Analyze results: ./analyze_results.sh"
echo ""
