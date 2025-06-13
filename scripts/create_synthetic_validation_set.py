#!/usr/bin/env python3
"""
Create synthetic validation dataset for molecular subtype testing.
This helps validate the model performance before EPOC data arrives.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import argparse
from tqdm import tqdm


def create_canonical_pattern(size=(224, 224)):
    """Create synthetic Canonical subtype pattern (E2F/MYC activation, sharp borders)"""
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Create sharp tumor nests with minimal infiltration
    for i in range(5):
        x = np.random.randint(20, size[0]-60)
        y = np.random.randint(20, size[1]-60)
        # Sharp-edged tumor regions
        draw.ellipse([x, y, x+40, y+40], fill=(200, 100, 150))
    
    # Add some structure
    for i in range(3):
        x1 = np.random.randint(0, size[0])
        y1 = np.random.randint(0, size[1])
        x2 = np.random.randint(0, size[0])
        y2 = np.random.randint(0, size[1])
        draw.line([x1, y1, x2, y2], fill=(180, 180, 200), width=2)
    
    # Apply minimal blur to maintain sharp interfaces
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img


def create_immune_pattern(size=(224, 224)):
    """Create synthetic Immune subtype pattern (band-like infiltration)"""
    img = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Central tumor region
    center_x, center_y = size[0]//2, size[1]//2
    draw.ellipse([center_x-40, center_y-40, center_x+40, center_y+40], 
                 fill=(200, 150, 170))
    
    # Band-like peritumoral infiltration
    for radius in range(50, 80, 5):
        for angle in np.linspace(0, 2*np.pi, 30):
            x = center_x + radius * np.cos(angle) + np.random.randint(-5, 5)
            y = center_y + radius * np.sin(angle) + np.random.randint(-5, 5)
            # Dense lymphocytes (blue/purple)
            draw.ellipse([x-3, y-3, x+3, y+3], fill=(100, 100, 200))
    
    # Immune highways extending inward
    for angle in np.linspace(0, 2*np.pi, 8):
        end_x = center_x + 90 * np.cos(angle)
        end_y = center_y + 90 * np.sin(angle)
        draw.line([center_x, center_y, end_x, end_y], 
                  fill=(120, 120, 220), width=3)
    
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    return img


def create_stromal_pattern(size=(224, 224)):
    """Create synthetic Stromal subtype pattern (EMT, fibrosis, immune exclusion)"""
    img = Image.new('RGB', size, color=(250, 240, 230))
    draw = ImageDraw.Draw(img)
    
    # Dense fibrotic patterns
    for i in range(15):
        x1 = np.random.randint(0, size[0])
        y1 = np.random.randint(0, size[1])
        x2 = x1 + np.random.randint(-50, 50)
        y2 = y1 + np.random.randint(-50, 50)
        # Wavy fibrotic strands
        draw.line([x1, y1, x2, y2], fill=(180, 150, 120), width=3)
    
    # Sparse tumor islands (immune excluded)
    for i in range(3):
        x = np.random.randint(30, size[0]-30)
        y = np.random.randint(30, size[1]-30)
        draw.ellipse([x, y, x+25, y+25], fill=(200, 130, 160))
        # Fibrotic barrier around tumor
        draw.ellipse([x-5, y-5, x+30, y+30], outline=(150, 120, 90), width=3)
    
    # Add texture
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    return img


def augment_image(img):
    """Apply random augmentations to increase diversity"""
    # Random rotation
    if np.random.random() > 0.5:
        angle = np.random.randint(-45, 45)
        img = img.rotate(angle, fillcolor=(255, 255, 255))
    
    # Random brightness/contrast
    if np.random.random() > 0.5:
        from PIL import ImageEnhance
        brightness = ImageEnhance.Brightness(img)
        img = brightness.enhance(np.random.uniform(0.8, 1.2))
        
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(np.random.uniform(0.8, 1.2))
    
    # Random flip
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    return img


def create_synthetic_dataset(output_dir, n_samples_per_class=100):
    """Create synthetic validation dataset"""
    output_path = Path(output_dir)
    
    # Create directories
    for subtype in ['canonical', 'immune', 'stromal']:
        (output_path / subtype).mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    print("Generating synthetic molecular subtype patterns...")
    
    # Canonical samples
    print(f"\nGenerating {n_samples_per_class} Canonical samples...")
    for i in tqdm(range(n_samples_per_class)):
        img = create_canonical_pattern()
        img = augment_image(img)
        img.save(output_path / 'Canonical' / f'canonical_{i:04d}.png')
    
    # Immune samples  
    print(f"\nGenerating {n_samples_per_class} Immune samples...")
    for i in tqdm(range(n_samples_per_class)):
        img = create_immune_pattern()
        img = augment_image(img)
        img.save(output_path / 'Immune' / f'immune_{i:04d}.png')
    
    # Stromal samples
    print(f"\nGenerating {n_samples_per_class} Stromal samples...")
    for i in tqdm(range(n_samples_per_class)):
        img = create_stromal_pattern()
        img = augment_image(img)
        img.save(output_path / 'Stromal' / f'stromal_{i:04d}.png')
    
    print(f"\n✅ Created {n_samples_per_class * 3} synthetic samples in {output_dir}")
    
    # Create summary file
    with open(output_path / 'dataset_info.txt', 'w') as f:
        f.write("Synthetic Molecular Subtype Validation Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {n_samples_per_class * 3}\n")
        f.write(f"- Canonical: {n_samples_per_class} samples\n")
        f.write(f"- Immune: {n_samples_per_class} samples\n")
        f.write(f"- Stromal: {n_samples_per_class} samples\n\n")
        f.write("Pattern characteristics:\n")
        f.write("- Canonical: Sharp tumor borders, minimal infiltration\n")
        f.write("- Immune: Band-like peritumoral infiltration, immune highways\n")
        f.write("- Stromal: Dense fibrosis, immune exclusion, sparse tumor islands\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic molecular subtype validation dataset")
    parser.add_argument('--output_dir', type=str, default='data/synthetic_validation',
                        help='Output directory for synthetic data')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Number of samples per subtype class')
    
    args = parser.parse_args()
    
    create_synthetic_dataset(args.output_dir, args.n_samples) 