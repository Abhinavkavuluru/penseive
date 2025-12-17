@echo off
REM Pensieve - Analyze Results
REM Compares performance of all ABR models

echo ================================================================================
echo Pensieve - Results Analysis
echo ================================================================================
echo.

cd /d "%~dp0..\..\test"

REM Check if results exist
if not exist results\log_sim_bb_trace_10mbps (
    echo ERROR: No results found!
    echo Please run the models first:
    echo   - run_bb.bat
    echo   - run_mpc.bat
    echo   - run_dp.bat
    echo Or run all at once: run_all_models.bat
    echo.
    pause
    exit /b 1
)

echo Analyzing results...
echo.

REM Run Python analysis script
python -c "import sys; sys.path.append('.'); exec(open(r'%~dp0analyze_results.py').read())"

if errorlevel 1 (
    echo.
    echo ERROR: Analysis failed
    echo Make sure NumPy is installed: pip install numpy
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Analysis Complete!
echo ================================================================================
echo.
echo Key Takeaways:
echo   - Higher Total Reward = Better overall performance
echo   - DP shows theoretical upper bound (offline optimal)
echo   - Gap between DP and others = room for improvement
echo   - Fewer quality switches = smoother playback
echo   - Lower rebuffering = better user experience
echo.
pause
