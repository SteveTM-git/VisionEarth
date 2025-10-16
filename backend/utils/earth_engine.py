import ee
import folium
from datetime import datetime, timedelta
import os

class EarthEngineClient:
    def __init__(self, project_id='careful-compass-475306-s4'):
        """Initialize Earth Engine"""
        try:
            # Try without project first
            ee.Initialize()
            print("✅ Earth Engine initialized successfully")
        except Exception as e:
            # If that fails, try with project
            try:
                ee.Initialize(project=project_id)
                print("✅ Earth Engine initialized with project")
            except Exception as e2:
                print(f"❌ Error initializing Earth Engine: {e2}")
    
    def get_satellite_image(self, lat, lon, start_date, end_date, scale=30):
        """
        Get Landsat 8 satellite imagery for a location
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            scale: Resolution in meters (default 30m for Landsat)
        """
        # Define point of interest
        point = ee.Geometry.Point([lon, lat])
        
        # Get Landsat 8 imagery
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterDate(start_date, end_date) \
            .filterBounds(point) \
            .sort('CLOUD_COVER') \
            .first()
        
        # Select RGB bands
        rgb = collection.select(['SR_B4', 'SR_B3', 'SR_B2'])
        
        return rgb, point
    
    def get_deforestation_hotspots(self, region_coords, start_date, end_date):
        """
        Detect potential deforestation using NDVI change
        
        Args:
            region_coords: [[lon, lat], [lon, lat], ...] polygon coordinates
            start_date: Start date
            end_date: End date
        """
        region = ee.Geometry.Polygon(region_coords)
        
        # Get imagery before and after
        before = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .median()
        
        # Calculate NDVI (Normalized Difference Vegetation Index)
        nir = before.select('SR_B5')  # Near-infrared
        red = before.select('SR_B4')  # Red
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        
        return ndvi, region
    
    def visualize_on_map(self, image, center, zoom=10):
        """
        Create a folium map with the satellite image
        """
        # Create map
        map_obj = folium.Map(location=center, zoom_start=zoom)
        
        # Get map ID for visualization
        vis_params = {
            'min': 0,
            'max': 3000,
            'bands': ['SR_B4', 'SR_B3', 'SR_B2']
        }
        
        map_id = image.getMapId(vis_params)
        
        # Add tile layer
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            overlay=True,
            name='Satellite Image'
        ).add_to(map_obj)
        
        folium.LayerControl().add_to(map_obj)
        
        return map_obj


# Test function
def test_earth_engine():
    """Test Earth Engine connection with Amazon rainforest"""
    client = EarthEngineClient(project_id='careful-compass-475306-s4')
    
    # Amazon rainforest coordinates (Rondônia, Brazil - deforestation hotspot)
    lat, lon = -10.0, -63.0
    
    # Get recent imagery
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"📡 Fetching satellite data for ({lat}, {lon})")
    print(f"📅 Date range: {start_date} to {end_date}")
    
    try:
        image, point = client.get_satellite_image(lat, lon, start_date, end_date)
        
        # Create visualization
        map_obj = client.visualize_on_map(image, center=[lat, lon], zoom=10)
        
        # Save map
        output_path = '../data/test_map.html'
        os.makedirs('../data', exist_ok=True)
        map_obj.save(output_path)
        
        print(f"✅ Map saved to: {output_path}")
        print("🌎 Open the HTML file in your browser to view!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_earth_engine()