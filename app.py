#!/usr/bin/env python3
"""
CRC Analysis Platform - Launcher Script
This script launches the main application from the src/ directory
"""

import sys
import os
from pathlib import Path

def main():
    """Main function to launch the CRC Analysis Platform"""
    try:
        # Add src directory to Python path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))

        # Change to src directory so relative imports work correctly
        original_cwd = os.getcwd()
        os.chdir(src_path)

        # Import and execute the main app
        import app as crc_app
        
        # Run the main function from the app
        if hasattr(crc_app, 'main'):
            crc_app.main()
        else:
            print("Warning: No main function found in app module")
            
    except ImportError as e:
        print(f"Error importing app module: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)
    finally:
        # Restore original working directory
        try:
            os.chdir(original_cwd)
        except:
            pass

if __name__ == "__main__":
    main() 