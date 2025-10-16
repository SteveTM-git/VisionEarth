import ee
import requests
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# Initialize Earth Engine
ee.Initialize(project='careful-compass-475306-s4')

class DeforestationDatasetBuilder:
    def __init__(self, output_dir='../data/real_dataset'):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / 'images'
        self.masks_dir = self.output_dir / 'masks'
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def get_deforestation_locations(self):
        """Known deforestation hotspots"""
        return {
            'Amazon_Rondonia_1': (-10.5, -63.5),
            'Amazon_Rondonia_2': (-9.8, -62.8),
            'Amazon_Para_1': (-3.5, -52.0),
            'Amazon_Para_2': (-4.2, -51.5),
            'Amazon_Mato_Grosso_1': (-12.0, -55.5),
            'Amazon_Mato_Grosso_2': (-11.5, -56.0),
            'Borneo_Kalimantan_1': (-1.0, 114.0),
            'Borneo_Kalimantan_2': (-0.5, 113.5),
            'Borneo_Sabah_1': (5.0, 117.5),
            'Congo_Basin_1': (-2.0, 23.0),
            'Congo_Basin_2': (-1.5, 24.0),
            'Madagascar_East_1': (-18.5, 48.5),
            'Madagascar_East_2': (-17.8, 49.0),
            'Sumatra_1': (0.5, 101.5),
            'Sumatra_2': (1.0, 102.0),
        }
    
    def create_deforestation_mask(self, location, buffer_size=5000):
        """
        Create deforestation mask using Hansen Global Forest Change data
        """
        lon, lat = location
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_size).bounds()
        
        # Hansen Global Forest Change dataset
        # Loss year: 0 = no loss, 1-21 = year of loss (2001-2021)
        hansen = ee.Image('UMD/hansen/global_forest_change_2021_v1_9')
        
        # Get forest loss (any year from 2001-2021)
        forest_loss = hansen.select('lossyear').gt(0)
        
        # Get tree cover in 2000
        tree_cover_2000 = hansen.select('treecover2000').gt(30)  # >30% tree cover
        
        # Get recent gain
        forest_gain = hansen.select('gain')
        
        # Water bodies (using JRC Global Surface Water)
        water = ee.Image('JRC/GSW1_3/GlobalSurfaceWater').select('occurrence').gt(50)
        
        # Create multi-class mask
        # 0: No change (forest that remains)
        # 1: Deforestation (tree cover lost)
        # 2: Urban/gain (new development or forest gain)
        # 3: Water
        
        mask = ee.Image(0)  # Default: no change
        mask = mask.where(forest_loss, 1)  # Deforestation
        mask = mask.where(forest_gain, 2)  # Gain/Urban
        mask = mask.where(water, 3)  # Water
        
        return mask, region
    
    def download_image_and_mask(self, name, location, idx):
        """Download satellite image and corresponding mask"""
        try:
            lon, lat = location
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(5000).bounds()
            
            print(f"\n📡 Processing {name} ({idx})...")
            
            # Get recent Sentinel-2 image (last 2 years)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            
            image = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterBounds(point) \
                .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                .sort('CLOUDY_PIXEL_PERCENTAGE') \
                .first()
            
            if image is None:
                print(f"⚠️  No suitable image found for {name}")
                return False
            
            # Get RGB bands (Sentinel-2: B4=Red, B3=Green, B2=Blue)
            rgb = image.select(['B4', 'B3', 'B2'])
            
            # Create deforestation mask
            mask, _ = self.create_deforestation_mask(location)
            
            # Download RGB image
            print("  📥 Downloading RGB image...")
            rgb_url = rgb.getThumbURL({
                'region': region,
                'dimensions': 512,
                'format': 'png'
            })
            
            rgb_response = requests.get(rgb_url, timeout=60)
            if rgb_response.status_code != 200:
                print(f"  ❌ Failed to download RGB")
                return False
            
            # Save RGB image
            rgb_path = self.images_dir / f'{name}_{idx:03d}.png'
            with open(rgb_path, 'wb') as f:
                f.write(rgb_response.content)
            
            # Download mask
            print("  📥 Downloading mask...")
            mask_url = mask.getThumbURL({
                'region': region,
                'dimensions': 512,
                'format': 'png',
                'palette': ['000000', 'FF0000', '808080', '0000FF']  # Black, Red, Gray, Blue
            })
            
            mask_response = requests.get(mask_url, timeout=60)
            if mask_response.status_code != 200:
                print(f"  ❌ Failed to download mask")
                return False
            
            # Save mask (convert to grayscale class indices)
            mask_path = self.masks_dir / f'{name}_{idx:03d}.png'
            mask_img = cv2.imdecode(np.frombuffer(mask_response.content, np.uint8), cv2.IMREAD_COLOR)
            
            # Convert RGB mask to class indices
            # Red (255,0,0) -> 1, Gray (128,128,128) -> 2, Blue (0,0,255) -> 3, Black (0,0,0) -> 0
            mask_gray = np.zeros((mask_img.shape[0], mask_img.shape[1]), dtype=np.uint8)
            
            # Deforestation (red-ish)
            mask_gray[(mask_img[:,:,2] > 200) & (mask_img[:,:,0] < 100)] = 1
            
            # Urban/Gain (gray-ish)
            mask_gray[(mask_img[:,:,0] > 100) & (mask_img[:,:,0] < 200)] = 2
            
            # Water (blue-ish)
            mask_gray[(mask_img[:,:,0] > 200) & (mask_img[:,:,2] < 100)] = 3
            
            cv2.imwrite(str(mask_path), mask_gray)
            
            print(f"  ✅ Saved: {rgb_path.name} and {mask_path.name}")
            
            # Check mask distribution
            unique, counts = np.unique(mask_gray, return_counts=True)
            total = mask_gray.size
            print(f"  📊 Mask distribution:")
            for u, c in zip(unique, counts):
                class_names = {0: "No change", 1: "Deforestation", 2: "Urban", 3: "Water"}
                print(f"     Class {u} ({class_names.get(u, 'Unknown')}): {c/total*100:.1f}%")
            
            time.sleep(2)  # Rate limiting
            return True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    def build_dataset(self, samples_per_location=3):
        """Build the complete dataset"""
        locations = self.get_deforestation_locations()
        
        print("="*60)
        print("🌍 Building Real Deforestation Dataset")
        print("="*60)
        print(f"📍 Locations: {len(locations)}")
        print(f"🎯 Samples per location: {samples_per_location}")
        print(f"📊 Total expected samples: {len(locations) * samples_per_location}")
        print("="*60)
        
        successful = 0
        failed = 0
        
        for name, location in tqdm(locations.items(), desc="Overall Progress"):
            for idx in range(samples_per_location):
                if self.download_image_and_mask(name, location, idx):
                    successful += 1
                else:
                    failed += 1
        
        print("\n" + "="*60)
        print("🎉 Dataset Building Complete!")
        print("="*60)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Images saved to: {self.images_dir}")
        print(f"📁 Masks saved to: {self.masks_dir}")
        print("="*60)
        
        return successful, failed


def main():
    print("🌍 VisionEarth - Real Deforestation Dataset Builder")
    print("="*60)
    
    builder = DeforestationDatasetBuilder(output_dir='../data/real_dataset')
    
    # Build dataset with 2 samples per location (to save time)
    # Increase to 5-10 for better training
    successful, failed = builder.build_dataset(samples_per_location=2)
    
    if successful > 0:
        print("\n✅ Dataset ready!")
        print("\n🚀 Next steps:")
        print("   1. Review the downloaded images")
        print("   2. Run training: python3 train.py")
        print("   3. Test with real satellite images!")
    else:
        print("\n⚠️  No samples were downloaded successfully")
        print("Check your Earth Engine authentication and internet connection")


if __name__ == "__main__":
    main()