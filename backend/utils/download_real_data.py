import requests
import zipfile
from pathlib import Path
import os

def download_dataset():
    """Download real deforestation dataset"""
    
    print("🌍 Downloading real deforestation dataset...")
    print("📦 This may take a few minutes...")
    
    # Example: Use a public deforestation dataset
    # You can use datasets from Kaggle, like:
    # - Amazon Rainforest Dataset
    # - Planet: Understanding the Amazon from Space
    
    datasets_info = """
    📚 Real Deforestation Datasets to Download:
    
    1. Kaggle - Planet: Understanding the Amazon from Space
       https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
       
    2. TerraClass Amazon Dataset
       http://www.inpe.br/cra/projetos_pesquisas/dados_terraclass.php
       
    3. Sentinel-2 Deforestation Dataset
       https://zenodo.org/search?q=deforestation+sentinel
    
    4. Global Forest Change Dataset (Hansen et al.)
       https://glad.earthengine.app/view/global-forest-change
    
    📝 Steps:
    1. Download dataset from one of the above sources
    2. Extract to: backend/data/real_dataset/
    3. Organize as:
       - data/real_dataset/images/
       - data/real_dataset/masks/
    4. Run: python3 train.py
    """
    
    print(datasets_info)
    
    # Check if user already has data
    data_path = Path('../data/real_dataset')
    if data_path.exists():
        print(f"\n✅ Found dataset at {data_path}")
        images = list((data_path / 'images').glob('*'))
        masks = list((data_path / 'masks').glob('*'))
        print(f"📊 Images: {len(images)}, Masks: {len(masks)}")
    else:
        print(f"\n⚠️  No dataset found at {data_path}")
        print("Please download a dataset using the links above")

if __name__ == "__main__":
    download_dataset()