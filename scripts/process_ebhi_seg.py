#!/usr/bin/env python3
"""
Process EBHI-SEG Dataset for CRC Model Training
Handles the specific structure of the EBHI-SEG histopathological image dataset
"""

import os
import sys
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import random
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app import memory_config

class EBHISegProcessor:
    """Process EBHI-SEG dataset for CRC analysis"""
    
    # Map EBHI-SEG categories to CRC molecular subtypes
    CATEGORY_TO_SUBTYPE = {
        'Adenocarcinoma': 'Canonical',      # Most common, maps to Canonical
        'High-grade IN': 'Canonical',       # High-grade neoplasia
        'Low-grade IN': 'Stromal',          # Low-grade, more stromal involvement
        'Normal': 'Normal',                 # Keep as separate category
        'Polyp': 'Stromal',                # Polyps often have stromal components
        'Serrated adenoma': 'Immune'        # Serrated pathway, immune involvement
    }
    
    def __init__(self, source_path: str, output_path: str):
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.memory_config = memory_config
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for subtype in ['Canonical', 'Immune', 'Stromal', 'Normal']:
                (self.output_path / split / subtype).mkdir(parents=True, exist_ok=True)
        
        (self.output_path / 'metadata').mkdir(exist_ok=True)
        (self.output_path / 'masks').mkdir(exist_ok=True)
        
    def analyze_dataset(self) -> Dict:
        """Analyze the EBHI-SEG dataset structure"""
        print("Analyzing EBHI-SEG dataset...")
        
        dataset_info = {
            'categories': {},
            'total_images': 0,
            'total_masks': 0,
            'image_sizes': [],
            'subtype_mapping': self.CATEGORY_TO_SUBTYPE
        }
        
        # Analyze each category
        for category_dir in self.source_path.iterdir():
            if category_dir.is_dir() and category_dir.name not in ['.DS_Store', 'readme.md']:
                image_dir = category_dir / 'image'
                label_dir = category_dir / 'label'
                
                if image_dir.exists() and label_dir.exists():
                    images = list(image_dir.glob('*.png'))
                    labels = list(label_dir.glob('*.png'))
                    
                    dataset_info['categories'][category_dir.name] = {
                        'num_images': len(images),
                        'num_labels': len(labels),
                        'mapped_subtype': self.CATEGORY_TO_SUBTYPE.get(category_dir.name, 'Unknown')
                    }
                    
                    dataset_info['total_images'] += len(images)
                    dataset_info['total_masks'] += len(labels)
                    
                    # Sample image sizes
                    for img_path in images[:5]:  # Check first 5 images
                        try:
                            with Image.open(img_path) as img:
                                dataset_info['image_sizes'].append(img.size)
                        except Exception as e:
                            print(f"Error reading {img_path}: {e}")
        
        return dataset_info
    
    def preprocess_image(self, image_path: Path, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess a single image"""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize to target size
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for better contrast
        lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_enhanced = cv2.merge([l, a, b])
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
        
        return img_enhanced
    
    def process_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Process the entire dataset and create train/val/test splits"""
        print("\nProcessing EBHI-SEG dataset...")
        
        # Collect all image-label pairs
        all_data = []
        
        for category_dir in self.source_path.iterdir():
            if category_dir.is_dir() and category_dir.name not in ['.DS_Store', 'readme.md']:
                image_dir = category_dir / 'image'
                label_dir = category_dir / 'label'
                
                if image_dir.exists() and label_dir.exists():
                    category = category_dir.name
                    subtype = self.CATEGORY_TO_SUBTYPE.get(category, 'Unknown')
                    
                    # Get all images
                    images = sorted(list(image_dir.glob('*.png')))
                    
                    for img_path in images:
                        label_path = label_dir / img_path.name
                        if label_path.exists():
                            all_data.append({
                                'image_path': img_path,
                                'label_path': label_path,
                                'category': category,
                                'subtype': subtype,
                                'filename': img_path.name
                            })
        
        print(f"Total image-label pairs: {len(all_data)}")
        
        # Create splits
        df = pd.DataFrame(all_data)
        metadata = []
        
        # Process each subtype separately to ensure balanced splits
        for subtype in df['subtype'].unique():
            subtype_data = df[df['subtype'] == subtype]
            
            # Split data
            train_val, test = train_test_split(
                subtype_data, 
                test_size=1-train_ratio-val_ratio, 
                random_state=42
            )
            
            train, val = train_test_split(
                train_val, 
                test_size=val_ratio/(train_ratio+val_ratio), 
                random_state=42
            )
            
            # Process and save images
            for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
                print(f"\nProcessing {split_name} split for {subtype}: {len(split_data)} images")
                
                for _, row in tqdm(split_data.iterrows(), total=len(split_data), 
                                  desc=f"{split_name}/{subtype}"):
                    try:
                        # Process image
                        img = self.preprocess_image(row['image_path'])
                        
                        # Save processed image
                        output_filename = f"{row['category']}_{row['filename']}"
                        output_path = self.output_path / split_name / subtype / output_filename
                        cv2.imwrite(str(output_path), img)
                        
                        # Copy mask to masks directory
                        mask_output = self.output_path / 'masks' / split_name / subtype / output_filename
                        mask_output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(row['label_path'], mask_output)
                        
                        # Store metadata
                        metadata.append({
                            'split': split_name,
                            'subtype': subtype,
                            'category': row['category'],
                            'filename': output_filename,
                            'original_image': str(row['image_path']),
                            'original_mask': str(row['label_path']),
                            'processed_image': str(output_path),
                            'processed_mask': str(mask_output)
                        })
                        
                    except Exception as e:
                        print(f"Error processing {row['image_path']}: {e}")
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(self.output_path / 'metadata' / 'dataset_metadata.csv', index=False)
        
        # Generate summary statistics
        self._generate_summary(metadata_df)
        
        return metadata_df
    
    def _generate_summary(self, metadata_df: pd.DataFrame):
        """Generate summary statistics"""
        print("\n" + "="*50)
        print("Dataset Summary:")
        print("="*50)
        
        # Overall statistics
        print(f"\nTotal processed images: {len(metadata_df)}")
        
        # Split distribution
        print("\nSplit distribution:")
        split_counts = metadata_df['split'].value_counts()
        for split, count in split_counts.items():
            print(f"  {split}: {count} ({count/len(metadata_df)*100:.1f}%)")
        
        # Subtype distribution
        print("\nSubtype distribution:")
        subtype_counts = metadata_df['subtype'].value_counts()
        for subtype, count in subtype_counts.items():
            print(f"  {subtype}: {count} ({count/len(metadata_df)*100:.1f}%)")
        
        # Category to subtype mapping
        print("\nCategory to subtype mapping:")
        category_mapping = metadata_df.groupby(['category', 'subtype']).size()
        for (category, subtype), count in category_mapping.items():
            print(f"  {category} → {subtype}: {count} images")
        
        # Save summary
        summary = {
            'total_images': len(metadata_df),
            'split_distribution': split_counts.to_dict(),
            'subtype_distribution': subtype_counts.to_dict(),
            'category_mapping': {
                category: metadata_df[metadata_df['category'] == category]['subtype'].iloc[0]
                for category in metadata_df['category'].unique()
            }
        }
        
        with open(self.output_path / 'metadata' / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    """Main processing pipeline"""
    
    # Paths
    source_path = "/Users/vijaythakore/Downloads/EBHI-SEG"
    project_path = Path("/Users/vijaythakore/Downloads/Downloads/Projects/CRC_Analysis_Project")
    output_path = project_path / "data" / "ebhi_seg_processed"
    
    # Initialize processor
    processor = EBHISegProcessor(source_path, output_path)
    
    # Analyze dataset
    dataset_info = processor.analyze_dataset()
    print("\nDataset Analysis:")
    print(f"Total images: {dataset_info['total_images']}")
    print(f"Total masks: {dataset_info['total_masks']}")
    print("\nCategories:")
    for category, info in dataset_info['categories'].items():
        print(f"  {category}: {info['num_images']} images → {info['mapped_subtype']}")
    
    # Save analysis
    with open(output_path / 'metadata' / 'dataset_analysis.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Process dataset
    metadata_df = processor.process_dataset(train_ratio=0.7, val_ratio=0.15)
    
    print("\n" + "="*50)
    print("EBHI-SEG processing complete!")
    print(f"Processed data saved to: {output_path}")
    print("\nNext steps:")
    print("1. Run: python scripts/train_model_with_ebhi.py")
    print("2. The trained model will be ready for EPOC validation")
    print("3. Integration with the CRC unified platform")

if __name__ == "__main__":
    main() 