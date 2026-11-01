@echo off
cd /d "%~dp0"
echo ========================================================
echo   PUSHING HYPOGUARDIAN AI UPDATES TO GITHUB
echo ========================================================
echo.
git push origin main
echo.
echo If the push succeeded, Streamlit Cloud will update automatically.
pause
