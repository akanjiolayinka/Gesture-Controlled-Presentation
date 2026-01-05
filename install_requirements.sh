#!/bin/bash
echo "Installing Gesture Presentation System Dependencies..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed!"
    echo "Please install Python 3.7+"
    exit 1
fi

echo "Creating/using local virtual environment (.venv)..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing required packages from requirements.txt..."
python -m pip install -r requirements.txt

echo
echo "Installation complete!"
echo "Run the system with: python gesture_presentation.py"
