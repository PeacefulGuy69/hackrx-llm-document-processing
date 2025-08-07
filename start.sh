#!/bin/bash

echo "Starting LLM Document Processing System..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo
    echo "⚠️  IMPORTANT: Please edit .env file and add your OpenAI API key!"
    echo "   Open .env and set OPENAI_API_KEY=your_api_key_here"
    echo
    read -p "Press enter to continue..."
fi

# Start the server
echo "Starting FastAPI server..."
python main.py
