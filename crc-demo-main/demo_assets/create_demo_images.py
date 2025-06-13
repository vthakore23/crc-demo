import numpy as np
from PIL import Image
import os

def create_pathology_sample(tissue_type, filename):
    """Create a simple synthetic pathology image"""
    
    # Image size
    width, height = 512, 512
    
    # Color schemes for different tissue types (H&E staining colors)
    color_schemes = {
        'tumor': {
            'base': (220, 180, 210),  # Pink base
            'nuclei': (100, 50, 120),  # Purple nuclei
            'density': 0.15
        },
        'stroma': {
            'base': (240, 200, 220),  # Light pink
            'nuclei': (120, 80, 140),  # Light purple
            'density': 0.05
        },
        'lymphocytes': {
            'base': (180, 140, 190),  # Purple-pink
            'nuclei': (80, 40, 100),   # Dark purple
            'density': 0.25
        },
        'mucosa': {
            'base': (250, 220, 240),  # Very light pink
            'nuclei': (140, 100, 160),  # Medium purple
            'density': 0.08
        },
        'complex': {
            'base': (210, 170, 200),  # Mixed pink
            'nuclei': (110, 60, 130),  # Mixed purple
            'density': 0.12
        }
    }
    
    scheme = color_schemes.get(tissue_type, color_schemes['tumor'])
    
    # Create base image with noise
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill with base color and add variation
    for i in range(3):
        base_val = scheme['base'][i]
        noise = np.random.normal(0, 15, (height, width))
        img_array[:, :, i] = np.clip(base_val + noise, 0, 255).astype(np.uint8)
    
    # Add cellular structures (nuclei)
    num_cells = int(width * height * scheme['density'] / 100)
    
    for _ in range(num_cells):
        # Random position
        x = np.random.randint(10, width - 10)
        y = np.random.randint(10, height - 10)
        
        # Random size
        size = np.random.randint(4, 8)
        
        # Draw circular nucleus
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                if dx*dx + dy*dy <= size*size:
                    py = y + dy
                    px = x + dx
                    if 0 <= py < height and 0 <= px < width:
                        # Add some variation to nucleus color
                        color_var = np.random.randint(-10, 10, 3)
                        nucleus_color = np.array(scheme['nuclei']) + color_var
                        img_array[py, px] = np.clip(nucleus_color, 0, 255)
    
    # Add some texture patterns
    if tissue_type == 'stroma':
        # Add fibrous patterns
        for _ in range(30):
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            length = np.random.randint(20, 60)
            angle = np.random.uniform(0, 2 * np.pi)
            
            for t in range(length):
                x = int(start_x + t * np.cos(angle))
                y = int(start_y + t * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    # Lighter fibrous tissue
                    fiber_color = np.array(scheme['base']) + 10
                    img_array[y, x] = np.clip(fiber_color, 0, 255)
    
    # Create PIL Image
    img = Image.fromarray(img_array)
    
    # Save the image
    img.save(filename, 'JPEG', quality=95)
    print(f"Created {filename}")

def main():
    """Generate demo pathology images"""
    
    samples = [
        ('tumor_sample.jpg', 'tumor'),
        ('stroma_sample.jpg', 'stroma'),
        ('lymphocytes_sample.jpg', 'lymphocytes'),
        ('mucosa_sample.jpg', 'mucosa'),
        ('complex_stroma_sample.jpg', 'complex'),
    ]
    
    for filename, tissue_type in samples:
        create_pathology_sample(tissue_type, filename)
    
    print("\nAll demo pathology images created successfully!")
    print("These are synthetic images for demonstration purposes.")

if __name__ == "__main__":
    main() 