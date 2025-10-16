import ee
import requests
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time

ee.Initialize(project='careful-compass-475306-s4')

def download_better_images():
    """Download images with better parameters"""
    output_dir = Path('../data/real_dataset_fixed')
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Focus on locations that worked + add more diverse ones
    locations = {
        'Borneo_Sabah': (5.0, 117.5),
        'Amazon_Rondonia': (-10.5, -63.5),
        'Amazon_Para': (-3.5, -52.0),
        'Sumatra_Aceh': (3.5, 97.5),
        'Madagascar': (-18.5, 48.5),
        'Brazil_Cerrado': (-12.5, -47.5),
        'Indonesia_Papua': (-3.5, 139.0),
        'Colombia_Amazon': (-1.5, -71.0),
    }
    
    successful = 0
    
    for name, (lat, lon) in tqdm(locations.items(), desc="Downloading"):
        for attempt in range(3):  # 3 attempts per location
            try:
                print(f"\n📡 {name} (attempt {attempt + 1})...")
                
                point = ee.Geometry.Point([lon, lat])
                region = point.buffer(5000).bounds()
                
                # Get Sentinel-2 image with better filtering
                collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                    .filterBounds(point) \
                    .filterDate('2023-01-01', '2024-12-31') \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE')
                
                # Try to get an image
                image = collection.first()
                
                # Visualization parameters for RGB
                vis_params = {
                    'min': 0,
                    'max': 3000,
                    'bands': ['B4', 'B3', 'B2']
                }
                
                # Download using getThumbURL (more reliable)
                url = image.getThumbURL({
                    'region': region.getInfo(),
                    'dimensions': 512,
                    'format': 'png',
                    **vis_params
                })
                
                print(f"  📥 Downloading from: {url[:80]}...")
                response = requests.get(url, timeout=60)
                
                if response.status_code != 200:
                    print(f"  ❌ HTTP {response.status_code}")
                    continue
                
                # Decode image
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None or img.mean() > 250:
                    print(f"  ❌ Invalid/white image")
                    continue
                
                # Save image
                img_path = images_dir / f'{name}_{attempt:03d}.png'
                cv2.imwrite(str(img_path), img)
                
                # Create mask using Hansen dataset
                hansen = ee.Image('UMD/hansen/global_forest_change_2021_v1_9')
                forest_loss = hansen.select('lossyear').gt(0)
                tree_cover = hansen.select('treecover2000').gt(30)
                water = ee.Image('JRC/GSW1_3/GlobalSurfaceWater').select('occurrence').gt(50)
                
                # Create mask
                mask_img = ee.Image(0)
                mask_img = mask_img.where(forest_loss, 1)
                mask_img = mask_img.where(water, 3)
                
                # Download mask
                mask_url = mask_img.getThumbURL({
                    'region': region.getInfo(),
                    'dimensions': 512,
                    'format': 'png',
                    'palette': ['000000', 'FF0000', '808080', '0000FF']
                })
                
                mask_response = requests.get(mask_url, timeout=60)
                mask_array = np.frombuffer(mask_response.content, np.uint8)
                mask_rgb = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
                
                # Convert to grayscale classes
                mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
                mask[(mask_rgb[:,:,2] > 200) & (mask_rgb[:,:,0] < 100)] = 1  # Deforestation
                mask[(mask_rgb[:,:,0] > 200) & (mask_rgb[:,:,2] < 100)] = 3  # Water
                
                mask_path = masks_dir / f'{name}_{attempt:03d}.png'
                cv2.imwrite(str(mask_path), mask)
                
                print(f"  ✅ Saved! (mean: {img.mean():.1f})")
                successful += 1
                
                time.sleep(2)
                break  # Success, move to next location
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                time.sleep(2)
                continue
    
    print(f"\n🎉 Downloaded {successful} valid images!")
    print(f"📁 Saved to: {output_dir}")
    
    # Update symlink or move files
    if successful > 5:
        print("\n💡 Updating dataset...")
        import shutil
        real_dataset = Path('../data/real_dataset')
        
        # Backup old dataset
        if real_dataset.exists():
            shutil.move(str(real_dataset), str(real_dataset.parent / 'real_dataset_backup'))
        
        # Use new dataset
        shutil.move(str(output_dir), str(real_dataset))
        print("✅ Dataset updated!")

if __name__ == "__main__":
    download_better_images()