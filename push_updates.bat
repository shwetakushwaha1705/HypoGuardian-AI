@echo off
cd /d "%~dp0"
echo ========================================================
echo   SYNCING AND PUSHING HYPOGUARDIAN AI TO GITHUB
echo ========================================================
echo.
git push -u origin main --force
echo.
echo ========================================================
echo  All files updated! Streamlit Cloud will now reload.
echo ========================================================
pause
