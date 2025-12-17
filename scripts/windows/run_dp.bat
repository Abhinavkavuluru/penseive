@echo off
REM Pensieve - Run Dynamic Programming (DP) Model
REM Computes offline-optimal solution with perfect future knowledge

echo ================================================================================
echo Pensieve - Dynamic Programming (DP) - Offline Optimal
echo ================================================================================
echo.
echo ABOUT THIS MODEL:
echo   - Computes the THEORETICAL OPTIMAL solution
echo   - Uses dynamic programming algorithm
echo   - Assumes PERFECT knowledge of future bandwidth
echo   - This is the upper bound - not achievable in real-time
echo   - Used as benchmark to evaluate other algorithms
echo.
echo EXECUTION TIME: ~5-10 minutes (much slower than other models)
echo.
echo NOTE: Requires C++ compiler (g++ or MSVC)
echo.
echo ================================================================================
echo.

REM Navigate to test directory
cd /d "%~dp0..\..\test"

REM Check if we're in the right directory
if not exist dp.cc (
    echo ERROR: Cannot find dp.cc
    echo Make sure you're running this from the pensieve repository
    pause
    exit /b 1
)

echo Step 1: Compiling C++ code...
echo.

REM Try g++ first (MinGW)
g++ --version >nul 2>&1
if not errorlevel 1 (
    echo Using g++ compiler...
    g++ -std=c++11 -O3 dp.cc -o dp.exe
    if errorlevel 1 (
        echo ERROR: Compilation failed with g++
        pause
        exit /b 1
    )
    echo Compilation successful!
    goto :run
)

REM Try cl (MSVC)
cl >nul 2>&1
if not errorlevel 1 (
    echo Using MSVC compiler...
    cl /EHsc /O2 /std:c++14 dp.cc /Fe:dp.exe
    if errorlevel 1 (
        echo ERROR: Compilation failed with MSVC
        pause
        exit /b 1
    )
    echo Compilation successful!
    goto :run
)

REM No compiler found
echo ERROR: No C++ compiler found!
echo.
echo Please install one of the following:
echo   1. MinGW-w64: https://www.mingw-w64.org/
echo   2. Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/
echo.
pause
exit /b 1

:run
echo.
echo ================================================================================
echo Step 2: Running DP algorithm...
echo ================================================================================
echo.
echo Network traces being tested:
echo   - trace_10mbps (high bandwidth)
echo   - trace_5mbps (medium bandwidth)
echo   - trace_variable (varying bandwidth)
echo.
echo This will take several minutes. Progress:
echo.

REM Run the model
dp.exe

echo.
echo ================================================================================
echo DP Model Completed!
echo ================================================================================
echo.
echo Results saved to: test\results\
echo   - log_sim_dp_trace_10mbps
echo   - log_sim_dp_trace_5mbps
echo   - log_sim_dp_trace_variable
echo.
echo DP output format (different from other models):
echo   First line: OPTIMAL total reward
echo   Following lines: Optimal decision sequence (reverse order)
echo.
echo This is the THEORETICAL UPPER BOUND:
echo   - Shows best possible performance with perfect knowledge
echo   - Real-time algorithms (BB, MPC, RL) should approach this
echo   - Gap between DP and real-time = room for improvement
echo.
echo Next: Run analyze_results.bat to see how BB and MPC compare to optimal
echo.
pause
