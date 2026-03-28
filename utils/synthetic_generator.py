"""
OncoSense - Synthetic Histopathology Image Generator
Generates realistic synthetic H&E-stained tissue images for demo training.
Malignant images have: higher cell density, larger irregular nuclei, less organized tissue.
Benign images have: lower cell density, smaller uniform nuclei, organized tubular structures.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random


def generate_synthetic_dataset(n_benign=150, n_malignant=150, img_size=224, seed=42):
    """
    Generate synthetic histopathology images for demo training.
    
    These images mimic H&E (Hematoxylin & Eosin) stained breast tissue slides:
    - Pink/purple base color (eosin stains cytoplasm pink, hematoxylin stains nuclei blue/purple)
    - Dark purple dots represent cell nuclei
    - Circular outlines represent tubular structures (glands)
    
    Malignant tissue characteristics:
    - High nuclear density (many cells, rapid proliferation)
    - Large, irregular nuclei (nuclear pleomorphism)
    - Fewer organized structures (loss of tissue architecture)
    - Darker overall staining
    
    Benign tissue characteristics:
    - Lower nuclear density (normal cell count)
    - Small, uniform nuclei
    - Many organized tubular structures (maintained architecture)
    - Lighter staining
    
    Returns:
        images: list of PIL Image objects
        labels: numpy array (0=malignant, 1=benign)
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)
    
    images = []
    labels = []
    
    # Generate benign images
    for i in range(n_benign):
        img = _generate_benign_image(img_size, rng, i)
        images.append(img)
        labels.append(1)  # 1 = benign
    
    # Generate malignant images
    for i in range(n_malignant):
        img = _generate_malignant_image(img_size, rng, i)
        images.append(img)
        labels.append(0)  # 0 = malignant
    
    return images, np.array(labels)


def _generate_benign_image(size, rng, idx):
    """Generate a synthetic benign tissue image."""
    # Light pink base (healthy tissue)
    base_r = rng.randint(230, 250)
    base_g = rng.randint(200, 225)
    base_b = rng.randint(210, 235)
    
    img_array = np.clip(
        np.array([base_r, base_g, base_b]) + rng.randn(size, size, 3) * 12,
        0, 255
    ).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # Many organized tubular structures (glands) — key benign indicator
    n_tubules = rng.randint(10, 20)
    for _ in range(n_tubules):
        x = rng.randint(20, size - 20)
        y = rng.randint(20, size - 20)
        rx = rng.randint(12, 35)
        ry = rng.randint(12, 35)
        outline_color = (
            rng.randint(190, 215),
            rng.randint(155, 180),
            rng.randint(175, 200)
        )
        draw.ellipse([x-rx, y-ry, x+rx, y+ry], outline=outline_color, width=2)
        
        # Small uniform nuclei along tubule walls
        n_wall_nuclei = rng.randint(4, 10)
        for j in range(n_wall_nuclei):
            angle = 2 * np.pi * j / n_wall_nuclei + rng.randn() * 0.3
            nx = int(x + rx * np.cos(angle)) + rng.randint(-3, 3)
            ny = int(y + ry * np.sin(angle)) + rng.randint(-3, 3)
            r = rng.randint(2, 4)  # Small uniform nuclei
            nuc_color = (rng.randint(80, 130), rng.randint(50, 90), rng.randint(120, 170))
            if 0 < nx < size and 0 < ny < size:
                draw.ellipse([nx-r, ny-r, nx+r, ny+r], fill=nuc_color)
    
    # Scattered stromal nuclei (low density)
    n_nuclei = rng.randint(40, 80)
    for _ in range(n_nuclei):
        x = rng.randint(5, size - 5)
        y = rng.randint(5, size - 5)
        r = rng.randint(2, 4)
        nuc_color = (rng.randint(90, 140), rng.randint(50, 100), rng.randint(130, 175))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=nuc_color)
    
    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    
    return img


def _generate_malignant_image(size, rng, idx):
    """Generate a synthetic malignant tissue image."""
    # Darker pink base (denser tissue, more hematoxylin uptake)
    base_r = rng.randint(205, 230)
    base_g = rng.randint(165, 195)
    base_b = rng.randint(185, 210)
    
    img_array = np.clip(
        np.array([base_r, base_g, base_b]) + rng.randn(size, size, 3) * 18,
        0, 255
    ).astype(np.uint8)
    
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    # Few or no organized structures (loss of architecture)
    n_structures = rng.randint(2, 6)
    for _ in range(n_structures):
        x = rng.randint(20, size - 20)
        y = rng.randint(20, size - 20)
        rx = rng.randint(8, 20)
        ry = rng.randint(8, 20)
        outline_color = (
            rng.randint(180, 205),
            rng.randint(140, 170),
            rng.randint(165, 190)
        )
        draw.ellipse([x-rx, y-ry, x+rx, y+ry], outline=outline_color, width=1)
    
    # Dense, large, irregular nuclei (key malignant indicators)
    n_nuclei = rng.randint(200, 350)
    for _ in range(n_nuclei):
        x = rng.randint(5, size - 5)
        y = rng.randint(5, size - 5)
        r = rng.randint(3, 9)  # Large, variable size (nuclear pleomorphism)
        
        # Darker nuclei (hyperchromatic)
        nuc_color = (rng.randint(60, 130), rng.randint(30, 80), rng.randint(100, 170))
        
        # Irregular shapes — use slightly different radii
        rx = r + rng.randint(-2, 2)
        ry = r + rng.randint(-2, 2)
        draw.ellipse([x-rx, y-ry, x+rx, y+ry], fill=nuc_color)
    
    # Mitotic figures (dark dense dots — signs of rapid division)
    n_mitosis = rng.randint(3, 8)
    for _ in range(n_mitosis):
        x = rng.randint(10, size - 10)
        y = rng.randint(10, size - 10)
        r = rng.randint(4, 7)
        dark_color = (rng.randint(40, 70), rng.randint(20, 50), rng.randint(60, 100))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=dark_color)
    
    # Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.7))
    
    return img


def generate_preview_pair(img_size=224, seed=99):
    """Generate one benign and one malignant image for preview."""
    rng = np.random.RandomState(seed)
    benign = _generate_benign_image(img_size, rng, 0)
    malignant = _generate_malignant_image(img_size, rng, 0)
    return benign, malignant
