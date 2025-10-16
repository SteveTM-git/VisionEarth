import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def verify_images():
    """Check if images are valid"""
    images_dir = Path('../data/real_dataset/images')
    masks_dir = Path('../data/real_dataset/masks')
    
    images = list(images_dir.glob('*.png'))
    
    print(f"📊 Found {len(images)} images")
    print("\nChecking image validity...\n")
    
    valid_count = 0
    white_count = 0
    
    for img_path in images:
        # Load image
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"❌ {img_path.name}: Cannot read")
            continue
        
        # Check if image is mostly white
        mean_intensity = img.mean()
        
        if mean_intensity > 250:
            print(f"⚠️  {img_path.name}: Mostly white (mean: {mean_intensity:.1f})")
            white_count += 1
        else:
            print(f"✅ {img_path.name}: Valid (mean: {mean_intensity:.1f})")
            valid_count += 1
        
        # Check mask
        mask_path = masks_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            unique_values = np.unique(mask)
            print(f"   Mask classes: {unique_values}")
    
    print(f"\n📊 Summary:")
    print(f"   Valid images: {valid_count}")
    print(f"   White/invalid images: {white_count}")
    
    # Show a sample
    if valid_count > 0:
        print("\n🖼️  Showing first valid image...")
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None and img.mean() < 250:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                mask_path = masks_dir / img_path.name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(img_rgb)
                axes[0].set_title(f'Image: {img_path.name}')
                axes[0].axis('off')
                
                axes[1].imshow(mask, cmap='viridis')
                axes[1].set_title('Mask')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig('../data/sample_visualization.png', dpi=150, bbox_inches='tight')
                print(f"✅ Saved sample to: data/sample_visualization.png")
                break

if __name__ == "__main__":
    verify_images()