@echo off
REM Pensieve - Run Model Predictive Control (MPC) Model
REM Optimization-based ABR algorithm with bandwidth prediction

echo ================================================================================
echo Pensieve - Model Predictive Control (MPC) Algorithm
echo ================================================================================
echo.
echo ABOUT THIS MODEL:
echo   - Uses optimization to predict future bandwidth
echo   - Looks ahead 5 video chunks into the future
echo   - Solves optimization problem to maximize QoE
echo   - More sophisticated than Buffer-Based approach
echo   - Adapts to changing network conditions
echo.
echo EXECUTION TIME: ~2-3 minutes
echo.
echo ================================================================================
echo.

REM Navigate to test directory
cd /d "%~dp0..\..\test"

REM Check if we're in the right directory
if not exist mpc.py (
    echo ERROR: Cannot find mpc.py
    echo Make sure you're running this from the pensieve repository
    pause
    exit /b 1
)

echo Starting MPC simulation...
echo.
echo Network traces being tested:
echo   - trace_10mbps (high bandwidth)
echo   - trace_5mbps (medium bandwidth)
echo   - trace_variable (varying bandwidth)
echo.
echo Progress:

REM Run the model
python mpc.py

echo.
echo ================================================================================
echo MPC Model Completed!
echo ================================================================================
echo.
echo Results saved to: test\results\
echo   - log_sim_mpc_trace_10mbps
echo   - log_sim_mpc_trace_5mbps
echo   - log_sim_mpc_trace_variable
echo.
echo Each file contains per-chunk metrics:
echo   [timestamp] [bitrate] [buffer] [rebuffer] [chunk_size] [delay] [reward]
echo.
echo MPC typically achieves:
echo   - Higher average bitrate than Buffer-Based
echo   - Fewer quality switches (smoother playback)
echo   - Better overall QoE score
echo.
echo Next: Run analyze_results.bat to compare with other models
echo.
pause
