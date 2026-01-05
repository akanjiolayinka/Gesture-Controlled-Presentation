@echo off
echo Installing Gesture Presentation System Dependencies...
echo.

python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH!
    echo Please install Python 3.7+ from python.org
    pause
    exit /b 1
)

echo Creating/using local virtual environment (.venv)...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required packages from requirements.txt...
python -m pip install -r requirements.txt

echo.
echo Installation complete!
echo Run the system with: run.bat
pause
