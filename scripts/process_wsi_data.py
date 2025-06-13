#!/usr/bin/env python3
"""
Process Whole Slide Images (WSI) from EBHI-SEG dataset for CRC analysis
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

def extract_rar_file(rar_path, extract_to):
    """Extract RAR file using patool"""
    try:
        import patool
        print(f"Extracting {rar_path} to {extract_to}...")
        os.makedirs(extract_to, exist_ok=True)
        
        # Try using patool
        patool.extract_archive(rar_path, outdir=extract_to, verbosity=1)
        print("Extraction completed successfully!")
        return True
    except Exception as e:
        print(f"Patool extraction failed: {e}")
        
        # Try using system unrar if available
        try:
            subprocess.run(['unrar', 'x', '-y', rar_path, extract_to], check=True)
            print("Extraction completed using system unrar!")
            return True
        except:
            print("Failed to extract RAR file. Please install unrar manually.")
            print("On macOS without Homebrew, you can:")
            print("1. Download The Unarchiver from Mac App Store")
            print("2. Or manually extract the RAR file and run this script again")
            return False

def explore_extracted_data(extract_path):
    """Explore the structure of extracted WSI data"""
    structure = {}
    
    for root, dirs, files in os.walk(extract_path):
        rel_path = os.path.relpath(root, extract_path)
        
        # Count file types
        file_types = {}
        for file in files:
            ext = Path(file).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        if files or dirs:
            structure[rel_path] = {
                'subdirs': dirs,
                'file_types': file_types,
                'total_files': len(files)
            }
    
    return structure

def analyze_wsi_images(data_path, sample_size=10):
    """Analyze a sample of WSI images to understand their properties"""
    image_info = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    # Find all image files
    image_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(image_files)} image files")
    
    # Analyze a sample
    sample_files = image_files[:min(sample_size, len(image_files))]
    
    for img_path in tqdm(sample_files, desc="Analyzing images"):
        try:
            with Image.open(img_path) as img:
                info = {
                    'path': os.path.relpath(img_path, data_path),
                    'size': img.size,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size_mb': os.path.getsize(img_path) / (1024 * 1024)
                }
                image_info.append(info)
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
    
    return image_info

def main():
    # Paths
    rar_path = "/Users/vijaythakore/Downloads/EBHI-SEG.rar"
    project_path = "/Users/vijaythakore/Downloads/Downloads/Projects/CRC_Analysis_Project"
    extract_path = os.path.join(project_path, "data", "wsi_data", "EBHI-SEG")
    
    # Check if already extracted
    if os.path.exists(extract_path) and os.listdir(extract_path):
        print(f"Data already extracted at {extract_path}")
        user_input = input("Re-extract? (y/n): ").lower()
        if user_input != 'y':
            print("Using existing extracted data...")
        else:
            shutil.rmtree(extract_path)
            if not extract_rar_file(rar_path, extract_path):
                return
    else:
        # Extract RAR file
        if not extract_rar_file(rar_path, extract_path):
            return
    
    # Explore structure
    print("\n" + "="*50)
    print("Exploring extracted data structure...")
    structure = explore_extracted_data(extract_path)
    
    print("\nDirectory structure:")
    for path, info in sorted(structure.items()):
        if info['total_files'] > 0 or info['subdirs']:
            print(f"\n{path}:")
            if info['subdirs']:
                print(f"  Subdirectories: {', '.join(info['subdirs'])}")
            if info['file_types']:
                print(f"  File types: {info['file_types']}")
    
    # Analyze images
    print("\n" + "="*50)
    print("Analyzing sample images...")
    image_analysis = analyze_wsi_images(extract_path)
    
    if image_analysis:
        print("\nImage analysis results:")
        for i, info in enumerate(image_analysis[:5]):  # Show first 5
            print(f"\nImage {i+1}:")
            print(f"  Path: {info['path']}")
            print(f"  Dimensions: {info['size'][0]} x {info['size'][1]}")
            print(f"  Mode: {info['mode']}")
            print(f"  Format: {info['format']}")
            print(f"  File size: {info['file_size_mb']:.2f} MB")
    
    # Save analysis results
    results_path = os.path.join(project_path, "data", "wsi_data", "analysis_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'structure': structure,
            'image_analysis': image_analysis,
            'total_images': len([f for f in os.walk(extract_path) for f in f[2] if Path(f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']])
        }, f, indent=2)
    
    print(f"\nAnalysis results saved to: {results_path}")
    print("\nNext steps:")
    print("1. Review the extracted data structure")
    print("2. Prepare data preprocessing pipeline")
    print("3. Integrate with CRC analysis model")

if __name__ == "__main__":
    main() 