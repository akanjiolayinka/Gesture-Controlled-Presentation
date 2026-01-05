@echo off
echo ================================================
echo   Gesture Presentation Controller
echo ================================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Starting application...
python gesture_presentation.py
echo.
echo Application stopped.
pause
