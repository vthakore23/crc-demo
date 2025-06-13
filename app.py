#!/usr/bin/env python3
"""
CRC Analysis Platform - Launcher Script
This script launches the main application from the src/ directory
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main application
if __name__ == "__main__":
    from app import main
    main() 