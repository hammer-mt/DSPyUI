#!/bin/bash

# Exit on error
set -e

# Function to deactivate virtual environment and exit
cleanup() {
    echo "Cleaning up..."
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    echo "Exited DSPy UI."
    exit
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT

# Check for Python installation
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists, create if not, then activate
if [ ! -d "dspyui_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv dspyui_env
else
    echo "Virtual environment already exists."
fi
echo "Activating virtual environment..."
source dspyui_env/bin/activate

# Check for .env file and source it if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
else
    echo "No .env file found. Make sure to set any necessary environment variables manually."
fi

# Install required packages if not already installed
if ! pip freeze | grep -q "gradio\|dspy\|pandas\|openai\|anthropic"; then
    echo "Installing required packages..."
    pip install gradio dspy pandas openai anthropic
else
    echo "Required packages are already installed."
fi

# Check if the Python script exists
if [ ! -f dspy_interface.py ]; then
    echo "dspy_interface.py not found. Please make sure the file exists in the current directory."
    exit 1
fi

# Launch the Gradio app
echo "Launching DSPy UI..."
python dspy_interface.py

# Note: The cleanup function will be called automatically when the script exits