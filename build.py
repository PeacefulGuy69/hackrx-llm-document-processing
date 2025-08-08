#!/usr/bin/env python3
"""Build script to prepare deployment directory"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def create_dist():
    """Create dist directory with necessary files"""
    print("Creating dist directory...")
    
    # Remove existing dist directory
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    
    # Create new dist directory
    os.makedirs("dist", exist_ok=True)
    
    # Files and directories to copy
    items_to_copy = [
        "*.py",
        "api/",
        "netlify/", 
        "vector_store/",
        "requirements.txt",
        "*.json",
        "*.toml",
        "*.md",
        "Procfile",
        "runtime.txt"
    ]
    
    # Copy files
    for item in items_to_copy:
        if "*" in item:
            # Handle wildcard patterns
            import glob
            for file in glob.glob(item):
                if os.path.isfile(file):
                    shutil.copy2(file, "dist/")
        else:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.copytree(item, f"dist/{item}", dirs_exist_ok=True)
                else:
                    shutil.copy2(item, "dist/")
    
    print("✅ Dist directory created successfully")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print("✅ Dependencies installed")

def main():
    """Main build function"""
    print("🚀 Starting build process...")
    
    try:
        install_dependencies()
        create_dist()
        print("🎉 Build completed successfully!")
        
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
