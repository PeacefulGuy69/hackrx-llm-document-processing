#!/bin/bash
set -e

# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
pip3 install -r requirements.txt

echo "Build completed successfully"
