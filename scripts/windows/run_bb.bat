@echo off
REM Pensieve - Run Buffer-Based (BB) Model
REM Simple threshold-based ABR algorithm

echo ================================================================================
echo Pensieve - Buffer-Based (BB) Algorithm
echo ================================================================================
echo.
echo ABOUT THIS MODEL:
echo   - Uses simple buffer thresholds to select video quality
echo   - RESERVOIR threshold: 5 seconds
echo   - CUSHION threshold: 10 seconds
echo   - If buffer ^< 5s: choose lowest quality
echo   - If buffer ^> 15s: choose highest quality
echo   - Otherwise: choose intermediate quality
echo.
echo EXECUTION TIME: ~1-2 minutes
echo.
echo ================================================================================
echo.

REM Navigate to test directory
cd /d "%~dp0..\..\test"

REM Check if we're in the right directory
if not exist bb.py (
    echo ERROR: Cannot find bb.py
    echo Make sure you're running this from the pensieve repository
    pause
    exit /b 1
)

echo Starting Buffer-Based simulation...
echo.
echo Network traces being tested:
echo   - trace_10mbps (high bandwidth)
echo   - trace_5mbps (medium bandwidth)
echo   - trace_variable (varying bandwidth)
echo.
echo Progress:

REM Run the model
python bb.py

echo.
echo ================================================================================
echo Buffer-Based Model Completed!
echo ================================================================================
echo.
echo Results saved to: test\results\
echo   - log_sim_bb_trace_10mbps
echo   - log_sim_bb_trace_5mbps
echo   - log_sim_bb_trace_variable
echo.
echo Each file contains per-chunk metrics:
echo   [timestamp] [bitrate] [buffer] [rebuffer] [chunk_size] [delay] [reward]
echo.
echo Higher reward = better quality of experience
echo.
echo Next: Run analyze_results.bat to compare with other models
echo.
pause
