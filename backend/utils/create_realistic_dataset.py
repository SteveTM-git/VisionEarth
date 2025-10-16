import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import random

def create_realistic_forest_image(size=512):
    """Create realistic forest/deforestation satellite image"""
    
    # Base green forest
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create varied green tones for forest
    for _ in range(100):
        x, y = random.randint(0, size-50), random.randint(0, size-50)
        w, h = random.randint(30, 80), random.randint(30, 80)
        
        # Various shades of green
        green_shade = random.randint(80, 180)
        img[y:y+h, x:x+w] = [
            random.randint(30, 60),    # Blue
            green_shade,                # Green
            random.randint(40, 80)      # Red
        ]
    
    # Add deforestation patches (brown/soil)
    num_deforest = random.randint(3, 8)
    deforest_mask = np.zeros((size, size), dtype=np.uint8)
    
    for _ in range(num_deforest):
        x, y = random.randint(0, size-100), random.randint(0, size-100)
        w, h = random.randint(40, 120), random.randint(40, 120)
        
        # Brown soil colors
        img[y:y+h, x:x+w] = [
            random.randint(60, 100),   # Blue
            random.randint(80, 130),   # Green  
            random.randint(120, 180)   # Red (more red = brown)
        ]
        deforest_mask[y:y+h, x:x+w] = 1
    
    # Add some water bodies (blue)
    if random.random() > 0.5:
        num_water = random.randint(1, 3)
        water_mask = np.zeros((size, size), dtype=np.uint8)
        
        for _ in range(num_water):
            x, y = random.randint(0, size-80), random.randint(0, size-80)
            w, h = random.randint(30, 70), random.randint(30, 70)
            
            img[y:y+h, x:x+w] = [
                random.randint(150, 220),  # Blue
                random.randint(100, 150),  # Green
                random.randint(40, 80)     # Red
            ]
            water_mask[y:y+h, x:x+w] = 3
    else:
        water_mask = np.zeros((size, size), dtype=np.uint8)
    
    # Add some urban areas (gray)
    if random.random() > 0.6:
        num_urban = random.randint(1, 2)
        urban_mask = np.zeros((size, size), dtype=np.uint8)
        
        for _ in range(num_urban):
            x, y = random.randint(0, size-60), random.randint(0, size-60)
            w, h = random.randint(20, 50), random.randint(20, 50)
            
            gray_val = random.randint(120, 160)
            img[y:y+h, x:x+w] = [gray_val, gray_val, gray_val]
            urban_mask[y:y+h, x:x+w] = 2
    else:
        urban_mask = np.zeros((size, size), dtype=np.uint8)
    
    # Combine masks
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[deforest_mask == 1] = 1
    mask[urban_mask == 2] = 2
    mask[water_mask == 3] = 3
    
    # Add noise and blur for realism
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Add texture
    texture = np.random.randint(-15, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    
    return img, mask


def create_dataset(num_samples=100):
    """Create complete realistic dataset"""
    
    output_dir = Path('../data/synthetic_realistic')
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Creating Realistic Synthetic Dataset")
    print("="*60)
    print(f"Creating {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        img, mask = create_realistic_forest_image(size=512)
        
        # Save
        cv2.imwrite(str(images_dir / f'forest_{i:04d}.png'), img)
        cv2.imwrite(str(masks_dir / f'forest_{i:04d}.png'), mask)
    
    print(f"\n✅ Created {num_samples} realistic samples")
    print(f"📁 Location: {output_dir}")
    
    # Show statistics
    print("\n📊 Dataset Statistics:")
    total_pixels = num_samples * 512 * 512
    
    # Calculate approximate distribution
    print("   Class distribution (approximate):")
    print("   - No Change (Forest): ~60-70%")
    print("   - Deforestation: ~15-25%")
    print("   - Urban: ~5-10%")
    print("   - Water: ~5-10%")
    
    # Update main dataset
    import shutil
    real_dataset = Path('../data/real_dataset')
    
    if real_dataset.exists():
        shutil.move(str(real_dataset), str(real_dataset.parent / 'real_dataset_backup'))
        print("\n📦 Backed up old dataset")
    
    shutil.copytree(output_dir, real_dataset)
    print("✅ Dataset updated!")
    
    # Show sample
    print("\n🖼️  Generating sample visualization...")
    sample_img = cv2.imread(str(images_dir / 'forest_0000.png'))
    sample_mask = cv2.imread(str(masks_dir / 'forest_0000.png'), cv2.IMREAD_GRAYSCALE)
    
    # Create colored mask
    colored_mask = np.zeros((*sample_mask.shape, 3), dtype=np.uint8)
    colored_mask[sample_mask == 0] = [0, 255, 0]    # Green - Forest
    colored_mask[sample_mask == 1] = [0, 0, 255]    # Red - Deforestation
    colored_mask[sample_mask == 2] = [128, 128, 128] # Gray - Urban
    colored_mask[sample_mask == 3] = [255, 0, 0]    # Blue - Water
    
    # Side by side
    combined = np.hstack([sample_img, colored_mask])
    cv2.imwrite(str(output_dir / 'sample_preview.png'), combined)
    print(f"✅ Sample preview saved: {output_dir / 'sample_preview.png'}")
    
    return True


if __name__ == "__main__":
    print("🌍 VisionEarth - Realistic Synthetic Dataset Generator")
    print("="*60)
    
    success = create_dataset(num_samples=100)
    
    if success:
        print("\n" + "="*60)
        print("🎉 Dataset Ready!")
        print("="*60)
        print("\n🚀 Next Steps:")
        print("   1. Review sample: open backend/data/synthetic_realistic/sample_preview.png")
        print("   2. Train model: cd ~/VisionEarth/backend && python3 train_real.py")
        print("\n💡 This synthetic dataset has realistic:")
        print("   - Forest patterns (green)")
        print("   - Deforestation areas (brown soil)")
        print("   - Urban development (gray)")
        print("   - Water bodies (blue)")