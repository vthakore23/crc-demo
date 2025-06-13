#!/usr/bin/env python3
"""
Molecular Subtype Training Demonstration
Generates sample data and runs the complete training pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import argparse
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularTrainingDemo:
    """Demo class for molecular subtype training"""
    
    def __init__(self, demo_dir: str = "demo_molecular_data", num_samples: int = 200):
        self.demo_dir = Path(demo_dir)
        self.num_samples = num_samples
        
        # Create demo directory structure
        self.demo_dir.mkdir(exist_ok=True)
        self.images_dir = self.demo_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        logger.info(f"Demo setup in: {self.demo_dir}")
    
    def generate_synthetic_histopathology_images(self):
        """Generate synthetic histopathology images for each molecular subtype"""
        logger.info(f"Generating {self.num_samples} synthetic histopathology images...")
        
        image_data = []
        subtypes = ['Canonical', 'Immune', 'Stromal']
        
        for i in range(self.num_samples):
            # Randomly assign subtype with realistic distribution
            subtype = np.random.choice(subtypes, p=[0.4, 0.35, 0.25])
            patient_id = f"DEMO_{i+1:04d}"
            filename = f"{patient_id}_{subtype.lower()}_patch.png"
            
            # Generate image based on subtype characteristics
            image = self._create_subtype_specific_image(subtype, (224, 224))
            
            # Save image
            image_path = self.images_dir / filename
            image.save(image_path)
            
            # Record metadata
            image_data.append({
                'patient_id': patient_id,
                'image_path': filename,
                'filename': filename,
                'molecular_subtype': subtype,
                'generated_features': self._get_subtype_features(subtype)
            })
            
            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{self.num_samples} images")
        
        return image_data
    
    def _create_subtype_specific_image(self, subtype: str, size: tuple):
        """Create synthetic image with subtype-specific characteristics"""
        width, height = size
        
        # Base tissue-like texture
        image = Image.new('RGB', size, (220, 180, 190))  # Light pink base
        draw = ImageDraw.Draw(image)
        
        # Add random background texture
        for _ in range(200):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            color = (
                random.randint(200, 240),
                random.randint(160, 200),
                random.randint(170, 210)
            )
            draw.point((x, y), fill=color)
        
        if subtype == 'Canonical':
            # Sharp tumor borders, organized glands
            self._add_canonical_features(draw, width, height)
            
        elif subtype == 'Immune':
            # Dense lymphocyte infiltration, band-like pattern
            self._add_immune_features(draw, width, height)
            
        elif subtype == 'Stromal':
            # Extensive fibrosis, desmoplastic reaction
            self._add_stromal_features(draw, width, height)
        
        return image
    
    def _add_canonical_features(self, draw, width, height):
        """Add canonical subtype features"""
        # Organized tumor nests with sharp borders
        for _ in range(5):
            x1 = random.randint(20, width-60)
            y1 = random.randint(20, height-60)
            x2 = x1 + random.randint(30, 50)
            y2 = y1 + random.randint(30, 50)
            
            # Sharp-bordered tumor regions (darker purple)
            draw.rectangle([x1, y1, x2, y2], fill=(140, 100, 150), outline=(100, 70, 120), width=2)
            
            # Add glandular structures
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            draw.ellipse([center_x-5, center_y-5, center_x+5, center_y+5], fill=(200, 160, 180))
        
        # Minimal immune infiltration (few small blue dots)
        for _ in range(5):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(50, 50, 150))  # Blue lymphocytes
    
    def _add_immune_features(self, draw, width, height):
        """Add immune subtype features"""
        # Dense band-like lymphocyte infiltration
        band_y = random.randint(height//4, 3*height//4)
        band_height = random.randint(30, 60)
        
        # Create immune band
        draw.rectangle([0, band_y, width, band_y + band_height], 
                      fill=(80, 80, 180), outline=(60, 60, 160))
        
        # Dense lymphocyte infiltration throughout
        for _ in range(100):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            size = random.randint(1, 3)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=(40, 40, 140))
        
        # Tertiary lymphoid structures
        for _ in range(3):
            center_x = random.randint(30, width-30)
            center_y = random.randint(30, height-30)
            radius = random.randint(10, 20)
            draw.ellipse([center_x-radius, center_y-radius, 
                         center_x+radius, center_y+radius], 
                        fill=(60, 60, 160), outline=(40, 40, 120), width=2)
        
        # Minimal fibrosis
        for _ in range(10):
            x1 = random.randint(0, width-20)
            y1 = random.randint(0, height-5)
            x2 = x1 + random.randint(10, 20)
            y2 = y1 + 2
            draw.rectangle([x1, y1, x2, y2], fill=(160, 140, 160))
    
    def _add_stromal_features(self, draw, width, height):
        """Add stromal subtype features"""
        # Extensive fibrosis with collagen deposition
        for _ in range(20):
            x1 = random.randint(0, width-30)
            y1 = random.randint(0, height-10)
            x2 = x1 + random.randint(20, 30)
            y2 = y1 + random.randint(5, 10)
            draw.rectangle([x1, y1, x2, y2], fill=(160, 140, 160))  # Fibrotic areas
        
        # Desmoplastic reaction (thick fibrous bands)
        for _ in range(5):
            x1 = random.randint(0, width//2)
            y1 = random.randint(0, height-20)
            x2 = x1 + random.randint(width//2, width-x1)
            y2 = y1 + random.randint(8, 15)
            draw.rectangle([x1, y1, x2, y2], fill=(140, 120, 140), outline=(120, 100, 120), width=1)
        
        # Immune exclusion (few, peripherally located lymphocytes)
        for _ in range(8):
            # Place lymphocytes only at edges
            if random.choice([True, False]):
                x = random.randint(0, 20)  # Left edge
            else:
                x = random.randint(width-20, width-1)  # Right edge
            y = random.randint(0, height-1)
            draw.ellipse([x-2, y-2, x+2, y+2], fill=(50, 50, 150))
        
        # Tumor nests embedded in stroma
        for _ in range(3):
            x1 = random.randint(20, width-40)
            y1 = random.randint(20, height-40)
            x2 = x1 + random.randint(15, 25)
            y2 = y1 + random.randint(15, 25)
            draw.ellipse([x1, y1, x2, y2], fill=(130, 90, 140))  # Embedded tumor
    
    def _get_subtype_features(self, subtype: str):
        """Get characteristic features for each subtype"""
        if subtype == 'Canonical':
            return {
                'tumor_content': random.uniform(0.6, 0.8),
                'immune_infiltration': random.uniform(0.05, 0.15),
                'stromal_content': random.uniform(0.1, 0.25),
                'border_sharpness': random.uniform(0.7, 0.9),
                'glandular_organization': random.uniform(0.6, 0.8)
            }
        elif subtype == 'Immune':
            return {
                'tumor_content': random.uniform(0.3, 0.5),
                'immune_infiltration': random.uniform(0.4, 0.7),
                'stromal_content': random.uniform(0.1, 0.3),
                'lymphocyte_density': random.uniform(0.5, 0.8),
                'tertiary_lymphoid_structures': random.uniform(0.3, 0.7)
            }
        elif subtype == 'Stromal':
            return {
                'tumor_content': random.uniform(0.2, 0.4),
                'immune_infiltration': random.uniform(0.05, 0.2),
                'stromal_content': random.uniform(0.5, 0.7),
                'fibrosis_score': random.uniform(0.6, 0.9),
                'desmoplastic_reaction': random.uniform(0.5, 0.8)
            }
    
    def create_manifest(self, image_data):
        """Create training manifest CSV"""
        manifest_df = pd.DataFrame(image_data)
        
        # Add additional columns that might be useful
        manifest_df['split'] = np.random.choice(['train', 'val', 'test'], 
                                               size=len(manifest_df), 
                                               p=[0.7, 0.2, 0.1])
        
        manifest_df['scanner_type'] = np.random.choice(['Aperio', 'Leica', 'Hamamatsu'], 
                                                      size=len(manifest_df))
        
        manifest_df['stain_quality'] = np.random.uniform(0.7, 1.0, size=len(manifest_df))
        
        # Save manifest
        manifest_path = self.demo_dir / "training_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        
        logger.info(f"Manifest created: {manifest_path}")
        logger.info(f"Subtype distribution:")
        print(manifest_df['molecular_subtype'].value_counts())
        
        return manifest_path
    
    def run_demo_training(self, manifest_path):
        """Run the complete training pipeline on demo data"""
        logger.info("🚀 Starting demo training pipeline...")
        
        # Prepare training command
        train_cmd = [
            sys.executable, "run_complete_molecular_training.py",
            "--manifest", str(manifest_path),
            "--data_dir", str(self.images_dir),
            "--output_dir", str(self.demo_dir / "training_output"),
            "--epochs", "5",  # Short demo training
            "--batch_size", "8",
            "--backbone", "efficientnet_b0",  # Smaller model for demo
            "--create_dummy_labels"  # Use generated labels
        ]
        
        logger.info(f"Training command: {' '.join(train_cmd)}")
        
        try:
            # Run training
            result = subprocess.run(train_cmd, cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                logger.info("✅ Demo training completed successfully!")
                return True
            else:
                logger.error(f"❌ Demo training failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Demo training execution failed: {e}")
            return False
    
    def create_demo_dataset(self):
        """Create complete demo dataset"""
        logger.info("Creating molecular subtype demo dataset...")
        
        # Generate images
        image_data = self.generate_synthetic_histopathology_images()
        
        # Create manifest
        manifest_path = self.create_manifest(image_data)
        
        # Create README for demo
        self._create_demo_readme()
        
        logger.info(f"✅ Demo dataset created successfully!")
        logger.info(f"📁 Location: {self.demo_dir}")
        logger.info(f"📋 Manifest: {manifest_path}")
        logger.info(f"🖼️ Images: {len(image_data)} synthetic histopathology images")
        
        return manifest_path
    
    def _create_demo_readme(self):
        """Create README for demo dataset"""
        readme_content = f"""# Molecular Subtype Demo Dataset

This is a synthetic dataset created for demonstrating the molecular subtype classification system.

## Dataset Overview

- **Total Images**: {self.num_samples}
- **Molecular Subtypes**: Canonical, Immune, Stromal
- **Image Format**: PNG, 224x224 pixels
- **Features**: Synthetic histopathology patterns based on Pitroda et al. 2018

## Subtype Characteristics

### Canonical
- Sharp tumor borders
- Organized glandular structures
- Minimal immune infiltration
- High tumor content (60-80%)

### Immune
- Dense band-like lymphocyte infiltration
- Tertiary lymphoid structures
- High immune content (40-70%)
- MSI-independent immune activation patterns

### Stromal
- Extensive fibrosis and desmoplastic reaction
- Immune exclusion patterns
- High stromal content (50-70%)
- EMT/angiogenesis features

## Files

- `training_manifest.csv`: Complete dataset manifest
- `images/`: Directory containing synthetic histopathology images
- Each image filename indicates patient ID and subtype

## Usage

Run the demo training:
```bash
python demo_molecular_training.py --run_training
```

Or use with the main training pipeline:
```bash
python run_complete_molecular_training.py \\
    --manifest {self.demo_dir}/training_manifest.csv \\
    --data_dir {self.demo_dir}/images \\
    --epochs 10 \\
    --batch_size 8
```

## Note

This is synthetic data for demonstration purposes only. 
For clinical applications, use real histopathology data with validated molecular labels.
"""
        
        with open(self.demo_dir / "README.md", 'w') as f:
            f.write(readme_content)

def main():
    parser = argparse.ArgumentParser(description='Molecular Subtype Training Demo')
    parser.add_argument('--demo_dir', type=str, default='demo_molecular_data',
                       help='Directory for demo data')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='Number of synthetic images to generate')
    parser.add_argument('--run_training', action='store_true',
                       help='Run training pipeline after creating demo data')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test with minimal data')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.num_samples = 50
    
    print("="*60)
    print("🧬 MOLECULAR SUBTYPE TRAINING DEMONSTRATION")
    print("="*60)
    print(f"Demo Directory: {args.demo_dir}")
    print(f"Sample Count: {args.num_samples}")
    print(f"Run Training: {args.run_training}")
    print("="*60)
    
    # Create demo
    demo = MolecularTrainingDemo(args.demo_dir, args.num_samples)
    
    # Generate demo dataset
    manifest_path = demo.create_demo_dataset()
    
    # Run training if requested
    if args.run_training:
        success = demo.run_demo_training(manifest_path)
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print(f"📁 Check results in: {args.demo_dir}/training_output")
        else:
            print("\n❌ Demo training failed. Check logs for details.")
    else:
        print(f"\n✅ Demo dataset created: {args.demo_dir}")
        print("To run training, use: python demo_molecular_training.py --run_training")

if __name__ == "__main__":
    main() 