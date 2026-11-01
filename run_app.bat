@echo off
echo ========================================================
echo   HYPOGUARDIAN AI: STARTING WEB DASHBOARD
echo   B.Tech Capstone & Research Prototype
echo ========================================================
echo.

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not detected in your PATH!
    echo Please install Python 3.11+ and try again.
    pause
    exit /b 1
)

:: Run Streamlit application
echo Launching Streamlit application on http://localhost:8501 ...
python -m streamlit run app.py

pause
