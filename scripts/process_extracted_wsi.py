#!/usr/bin/env python3
"""
Process extracted Whole Slide Images (WSI) for CRC model training
This script assumes the EBHI-SEG data has been manually extracted
"""

import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import cv2
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.memory_config import MemoryConfig

class WSIProcessor:
    """Process WSI data for CRC analysis"""
    
    def __init__(self, data_path: str, output_path: str):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.memory_config = MemoryConfig()
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "processed").mkdir(exist_ok=True)
        (self.output_path / "patches").mkdir(exist_ok=True)
        (self.output_path / "metadata").mkdir(exist_ok=True)
        
    def explore_dataset(self) -> Dict:
        """Explore the dataset structure"""
        print("Exploring dataset structure...")
        
        structure = {
            'total_files': 0,
            'image_files': [],
            'annotation_files': [],
            'other_files': [],
            'subdirectories': {}
        }
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
        annotation_extensions = {'.xml', '.json', '.txt', '.csv'}
        
        for root, dirs, files in os.walk(self.data_path):
            rel_path = Path(root).relative_to(self.data_path)
            
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                if ext in image_extensions:
                    structure['image_files'].append(str(rel_path / file))
                elif ext in annotation_extensions:
                    structure['annotation_files'].append(str(rel_path / file))
                else:
                    structure['other_files'].append(str(rel_path / file))
                    
                structure['total_files'] += 1
        
        # Analyze subdirectory structure
        for subdir in self.data_path.iterdir():
            if subdir.is_dir():
                structure['subdirectories'][subdir.name] = {
                    'files': len(list(subdir.rglob('*'))),
                    'images': len([f for f in subdir.rglob('*') if f.suffix.lower() in image_extensions])
                }
        
        return structure
    
    def analyze_image_properties(self, sample_size: int = 20) -> List[Dict]:
        """Analyze properties of sample images"""
        print(f"\nAnalyzing image properties (sample size: {sample_size})...")
        
        image_files = [f for f in self.data_path.rglob('*') 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
        
        if not image_files:
            print("No image files found!")
            return []
        
        # Sample images
        sample_files = random.sample(image_files, min(sample_size, len(image_files)))
        
        image_properties = []
        for img_path in tqdm(sample_files, desc="Analyzing images"):
            try:
                # Get basic properties with PIL
                with Image.open(img_path) as img:
                    properties = {
                        'filename': img_path.name,
                        'relative_path': str(img_path.relative_to(self.data_path)),
                        'size': img.size,
                        'mode': img.mode,
                        'format': img.format,
                        'file_size_mb': img_path.stat().st_size / (1024 * 1024)
                    }
                
                # Get additional properties with OpenCV
                img_cv = cv2.imread(str(img_path))
                if img_cv is not None:
                    properties['shape'] = img_cv.shape
                    properties['dtype'] = str(img_cv.dtype)
                    properties['mean_intensity'] = float(np.mean(img_cv))
                    properties['std_intensity'] = float(np.std(img_cv))
                
                image_properties.append(properties)
                
            except Exception as e:
                print(f"Error analyzing {img_path}: {e}")
        
        return image_properties
    
    def extract_patches(self, image_path: Path, patch_size: int = 512, 
                       stride: int = 256, min_tissue_ratio: float = 0.5) -> List[np.ndarray]:
        """Extract patches from a WSI image"""
        patches = []
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return patches
            
            h, w = img.shape[:2]
            
            # Extract patches with sliding window
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = img[y:y+patch_size, x:x+patch_size]
                    
                    # Check tissue ratio (simple threshold-based)
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    tissue_mask = gray_patch < 220  # Assuming white background
                    tissue_ratio = np.sum(tissue_mask) / (patch_size * patch_size)
                    
                    if tissue_ratio >= min_tissue_ratio:
                        patches.append(patch)
            
        except Exception as e:
            print(f"Error extracting patches from {image_path}: {e}")
        
        return patches
    
    def prepare_training_data(self, patch_size: int = 512, max_patches_per_image: int = 50):
        """Prepare data for model training"""
        print("\nPreparing training data...")
        
        # Find all images
        image_files = [f for f in self.data_path.rglob('*') 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
        
        if not image_files:
            print("No image files found!")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Process images and extract patches
        all_patches = []
        metadata = []
        
        for img_path in tqdm(image_files[:10], desc="Processing images"):  # Process first 10 for demo
            patches = self.extract_patches(img_path, patch_size=patch_size)
            
            # Limit patches per image
            if len(patches) > max_patches_per_image:
                patches = random.sample(patches, max_patches_per_image)
            
            # Save patches
            for i, patch in enumerate(patches):
                patch_filename = f"{img_path.stem}_patch_{i:04d}.png"
                patch_path = self.output_path / "patches" / patch_filename
                cv2.imwrite(str(patch_path), patch)
                
                # Store metadata
                metadata.append({
                    'patch_filename': patch_filename,
                    'source_image': str(img_path.relative_to(self.data_path)),
                    'patch_index': i,
                    'patch_size': patch_size
                })
            
            all_patches.extend(patches)
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(self.output_path / "metadata" / "patch_metadata.csv", index=False)
        
        print(f"\nExtracted {len(all_patches)} patches from {len(image_files)} images")
        print(f"Patches saved to: {self.output_path / 'patches'}")
        
        return metadata_df
    
    def create_training_splits(self, metadata_df: pd.DataFrame, 
                             train_ratio: float = 0.7, val_ratio: float = 0.2):
        """Create train/val/test splits"""
        print("\nCreating training splits...")
        
        # Get unique source images
        unique_images = metadata_df['source_image'].unique()
        n_images = len(unique_images)
        
        # Shuffle and split
        np.random.shuffle(unique_images)
        
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        train_images = unique_images[:n_train]
        val_images = unique_images[n_train:n_train+n_val]
        test_images = unique_images[n_train+n_val:]
        
        # Assign splits
        metadata_df['split'] = 'test'
        metadata_df.loc[metadata_df['source_image'].isin(train_images), 'split'] = 'train'
        metadata_df.loc[metadata_df['source_image'].isin(val_images), 'split'] = 'val'
        
        # Save split information
        metadata_df.to_csv(self.output_path / "metadata" / "patch_metadata_with_splits.csv", index=False)
        
        print(f"Train: {len(metadata_df[metadata_df['split']=='train'])} patches")
        print(f"Val: {len(metadata_df[metadata_df['split']=='val'])} patches")
        print(f"Test: {len(metadata_df[metadata_df['split']=='test'])} patches")
        
        return metadata_df

def main():
    """Main processing pipeline"""
    
    # Paths
    project_path = Path("/Users/vijaythakore/Downloads/Downloads/Projects/CRC_Analysis_Project")
    wsi_data_path = project_path / "data" / "wsi_data" / "EBHI-SEG"
    output_path = project_path / "data" / "wsi_processed"
    
    # Check if data exists
    if not wsi_data_path.exists():
        print(f"WSI data not found at: {wsi_data_path}")
        print("\nPlease extract the EBHI-SEG.rar file to:")
        print(f"  {wsi_data_path}")
        print("\nYou can use The Unarchiver (Mac App Store) or any RAR extraction tool.")
        return
    
    # Initialize processor
    processor = WSIProcessor(wsi_data_path, output_path)
    
    # Explore dataset
    structure = processor.explore_dataset()
    print(f"\nDataset structure:")
    print(f"  Total files: {structure['total_files']}")
    print(f"  Image files: {len(structure['image_files'])}")
    print(f"  Annotation files: {len(structure['annotation_files'])}")
    print(f"  Subdirectories: {list(structure['subdirectories'].keys())}")
    
    # Save structure
    with open(output_path / "metadata" / "dataset_structure.json", 'w') as f:
        json.dump(structure, f, indent=2)
    
    # Analyze images
    image_properties = processor.analyze_image_properties()
    
    if image_properties:
        print(f"\nSample image properties:")
        for prop in image_properties[:3]:
            print(f"  {prop['filename']}: {prop['size'][0]}x{prop['size'][1]}, {prop['file_size_mb']:.2f} MB")
    
    # Save analysis
    with open(output_path / "metadata" / "image_analysis.json", 'w') as f:
        json.dump(image_properties, f, indent=2)
    
    # Prepare training data
    metadata_df = processor.prepare_training_data()
    
    if metadata_df is not None and not metadata_df.empty:
        # Create splits
        processor.create_training_splits(metadata_df)
        
        print("\n" + "="*50)
        print("WSI data processing complete!")
        print(f"Processed data saved to: {output_path}")
        print("\nNext steps:")
        print("1. Review the extracted patches and metadata")
        print("2. Integrate with CRC model training pipeline")
        print("3. Update model architecture to handle WSI patches")

if __name__ == "__main__":
    main() 