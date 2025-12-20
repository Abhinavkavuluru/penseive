@echo off
REM Pensieve - Run All Models
REM Executes BB, MPC, and DP models sequentially

echo ================================================================================
echo Pensieve - Run All ABR Models
echo ================================================================================
echo.
echo This script will run all three models:
echo   1. Buffer-Based (BB) - ~1-2 minutes
echo   2. Model Predictive Control (MPC) - ~2-3 minutes
echo   3. Dynamic Programming (DP) - ~5-10 minutes
echo.
echo Total estimated time: 10-15 minutes
echo.
echo Press any key to start, or Ctrl+C to cancel...
pause >nul
echo.

REM Record start time
echo Started at: %TIME%
echo.

echo ================================================================================
echo [1/3] Running Buffer-Based (BB) Model
echo ================================================================================
echo.

cd /d "%~dp0..\..\test"

if not exist bb.py (
    echo ERROR: Cannot find test directory
    pause
    exit /b 1
)

echo Running BB...
python bb.py
if errorlevel 1 (
    echo ERROR: BB model failed
    pause
    exit /b 1
)

echo.
echo BB completed successfully!
echo.
timeout /t 2 /nobreak >nul

echo ================================================================================
echo [2/3] Running Model Predictive Control (MPC)
echo ================================================================================
echo.

echo Running MPC...
python mpc.py
if errorlevel 1 (
    echo ERROR: MPC model failed
    pause
    exit /b 1
)

echo.
echo MPC completed successfully!
echo.
timeout /t 2 /nobreak >nul

echo ================================================================================
echo [3/3] Running Dynamic Programming (DP) - Offline Optimal
echo ================================================================================
echo.

REM Check for C++ compiler
g++ --version >nul 2>&1
if errorlevel 1 (
    cl >nul 2>&1
    if errorlevel 1 (
        echo WARNING: No C++ compiler found. Skipping DP model.
        echo Install MinGW or Visual Studio Build Tools to run DP.
        goto :results
    )
    echo Compiling DP with MSVC...
    cl /EHsc /O2 /std:c++14 dp.cc /Fe:dp.exe
) else (
    echo Compiling DP with g++...
    g++ -std=c++11 -O3 dp.cc -o dp.exe
)

if errorlevel 1 (
    echo WARNING: DP compilation failed. Skipping DP model.
    goto :results
)

echo Running DP...
dp.exe
if errorlevel 1 (
    echo ERROR: DP model failed
    pause
    exit /b 1
)

echo.
echo DP completed successfully!
echo.

:results
echo ================================================================================
echo All Models Completed!
echo ================================================================================
echo.
echo Finished at: %TIME%
echo.
echo Results location: test\results\
echo.
dir /b results\log_sim_*
echo.
echo ================================================================================
echo Next Steps:
echo ================================================================================
echo.
echo Run analyze_results.bat to see detailed performance comparison
echo.
pause

REM Navigate back to scripts directory
cd /d "%~dp0"

REM Optionally run analysis
choice /C YN /M "Would you like to analyze results now"
if errorlevel 2 goto :end
if errorlevel 1 call analyze_results.bat

:end
