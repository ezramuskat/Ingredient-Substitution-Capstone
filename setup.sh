#!/bin/bash

PYTHON_VERSION=3.11

# Assert that the correct Python version is available
if ! command -v python$PYTHON_VERSION &> /dev/null
then
    echo "Python $PYTHON_VERSION is not installed. Please install it before proceeding."
    exit
fi

# Create a virtual environment
VENV_DIR=.venv
python$PYTHON_VERSION -m venv $VENV_DIR

# Install dependencies
pip install -r requirements.txt

# Setup complete
echo "Setup complete! Run 'source $VENV_DIR/bin/activate' to activate your environment."