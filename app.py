#!/usr/bin/env python3
"""
CRC Analysis Platform - Launcher Script
This script launches the main application from the src/ directory
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Change to src directory so relative imports work correctly
os.chdir(src_path)

# Import and execute the main app
if __name__ == "__main__":
    # Import the app module which will execute when imported
    import app 