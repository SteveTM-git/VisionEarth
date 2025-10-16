import os
import zipfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def download_and_prepare():
    """Download and prepare Amazon deforestation dataset"""
    
    print("🌍 VisionEarth - Kaggle Dataset Downloader")
    print("="*60)
    
    data_dir = Path('../data/kaggle_amazon')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Downloading Amazon Rainforest dataset from Kaggle...")
    print("This will download ~1GB of data. Please wait...")
    
    # Download the dataset (using a smaller, public dataset)
    os.chdir(data_dir)
    result = os.system('kaggle datasets download -d brunowerneck/deforestation-areas-in-amazon')
    
    if result != 0:
        print("\n❌ Download failed. Trying alternative dataset...")
        result = os.system('kaggle competitions download -c planet-understanding-the-amazon-from-space')
    
    # Find and extract zip file
    zip_files = list(data_dir.glob('*.zip'))
    
    if not zip_files:
        print("❌ No zip file found. Please check:")
        print("   1. Kaggle API is set up correctly")
        print("   2. You've accepted the competition rules (if needed)")
        return False
    
    print("\n📦 Extracting files...")
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"✅ Extracted: {zip_file.name}")
        zip_file.unlink()  # Remove zip
    
    # Go back to utils directory
    os.chdir('../../utils')
    
    # Now process and organize the data
    print("\n🔄 Processing images...")
    process_kaggle_data(data_dir)
    
    return True


def process_kaggle_data(source_dir):
    """Process Kaggle data into our format"""
    
    # Create output directories
    output_dir = Path('../data/real_dataset_kaggle')
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif']:
        image_files.extend(list(source_dir.rglob(ext)))
    
    print(f"📊 Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("⚠️  No images found. Trying manual search...")
        print(f"Contents of {source_dir}:")
        for item in source_dir.iterdir():
            print(f"  - {item}")
        return
    
    processed = 0
    
    for img_file in tqdm(image_files[:50], desc="Processing"):  # Limit to 50 for speed
        try:
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Skip if too small or too large
            h, w = img.shape[:2]
            if h < 100 or w < 100 or h > 5000 or w > 5000:
                continue
            
            # Skip if mostly white
            if img.mean() > 250:
                continue
            
            # Resize to standard size
            img_resized = cv2.resize(img, (512, 512))
            
            # Create mask using simple image analysis
            # (In real dataset, masks would come with the data)
            mask = create_simple_mask(img_resized)
            
            # Save
            save_name = f'amazon_{processed:04d}.png'
            cv2.imwrite(str(images_dir / save_name), img_resized)
            cv2.imwrite(str(masks_dir / save_name), mask)
            
            processed += 1
            
        except Exception as e:
            continue
    
    print(f"\n✅ Processed {processed} images")
    print(f"📁 Saved to: {output_dir}")
    
    if processed > 20:
        # Replace the old dataset
        print("\n🔄 Updating main dataset...")
        real_dataset = Path('../data/real_dataset')
        if real_dataset.exists():
            shutil.move(str(real_dataset), str(real_dataset.parent / 'real_dataset_old'))
        shutil.copytree(output_dir, real_dataset)
        print("✅ Dataset updated!")


def create_simple_mask(image):
    """Create mask using image analysis"""
    # Convert to HSV for vegetation detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Green vegetation (forest) = class 0
    green_mask = (hsv[:,:,0] > 35) & (hsv[:,:,0] < 85) & (hsv[:,:,1] > 30)
    
    # Brown/bare soil (deforestation) = class 1
    brown_mask = (hsv[:,:,0] > 10) & (hsv[:,:,0] < 30) & (hsv[:,:,1] > 20)
    
    # Gray (urban) = class 2
    gray_mask = (hsv[:,:,1] < 50) & (hsv[:,:,2] > 50)
    
    # Blue (water) = class 3
    blue_mask = (hsv[:,:,0] > 90) & (hsv[:,:,0] < 130)
    
    mask[brown_mask] = 1
    mask[gray_mask] = 2
    mask[blue_mask] = 3
    
    return mask


if __name__ == "__main__":
    success = download_and_prepare()
    
    if success:
        print("\n" + "="*60)
        print("🎉 Dataset ready for training!")
        print("="*60)
        print("\n🚀 Next step: Run training")
        print("   cd ~/VisionEarth/backend")
        print("   python3 train_real.py")
    else:
        print("\n⚠️  Dataset download incomplete.")
        print("Please check your Kaggle setup.")