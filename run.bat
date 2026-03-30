@echo off
echo Running Quantum ML Healthcare Simulation...
python run_experiments.py
if %ERRORLEVEL% NEQ 0 (
    echo Simulation failed. Check error_log.txt for details.
    pause
    exit /b %ERRORLEVEL%
)
echo Simulation completed. Starting interactive dashboard...
streamlit run dashboard.py
pause
