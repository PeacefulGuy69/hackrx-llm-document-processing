#!/bin/bash
set -e

echo "Starting build process..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install requirements
pip3 install -r requirements.txt

# Create dist directory if it doesn't exist
mkdir -p dist

# Copy necessary files to dist
cp -r *.py dist/ 2>/dev/null || true
cp -r api/ dist/ 2>/dev/null || true
cp -r netlify/ dist/ 2>/dev/null || true
cp -r vector_store/ dist/ 2>/dev/null || true
cp requirements.txt dist/ 2>/dev/null || true
cp *.json dist/ 2>/dev/null || true
cp *.toml dist/ 2>/dev/null || true
cp *.md dist/ 2>/dev/null || true

echo "Build completed successfully - dist directory created"
