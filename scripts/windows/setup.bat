@echo off
REM Pensieve - Windows Setup Script
REM This script installs all required Python dependencies

echo ================================================================================
echo Pensieve - Setup Script for Windows
echo ================================================================================
echo.
echo This script will install the required Python packages.
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

echo Installing required packages...
echo.

REM Upgrade pip
echo [1/5] Upgrading pip...
python -m pip install --upgrade pip

REM Install NumPy
echo [2/5] Installing NumPy...
pip install numpy==1.26.4

REM Install TensorFlow with correct version
echo [3/5] Installing TensorFlow 2.14.0...
pip install tensorflow==2.14.0

REM Install ml_dtypes (critical for compatibility)
echo [4/5] Installing ml_dtypes 0.2.0...
pip install ml_dtypes==0.2.0

REM Install matplotlib for plotting
echo [5/5] Installing matplotlib...
pip install matplotlib

echo.
echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.

REM Verify installation
echo Verifying installation...
python -c "import numpy; print('NumPy version:', numpy.__version__)"
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"

echo.
echo Setup completed successfully!
echo.
echo Next steps:
echo   1. Run individual models: run_bb.bat, run_mpc.bat, run_dp.bat
echo   2. Run all models: run_all_models.bat
echo   3. Analyze results: analyze_results.bat
echo.
pause
