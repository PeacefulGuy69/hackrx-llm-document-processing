import sys
import os

# Add the parent directory to the Python path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# This is the entry point for Vercel
# Vercel will automatically detect this file and use it as the handler
