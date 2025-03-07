#!/bin/bash

PYTHON_VERSION=3.11

# Create a virtual environment
VENV_DIR=.venv
python$PYTHON_VERSION -m venv $VENV_DIR

# Install dependencies
pip install -r requirements.txt

# Setup complete
echo "Setup complete! Run 'source $VENV_DIR/bin/activate' to activate your environment."