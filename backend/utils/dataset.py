import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SatelliteDataset(Dataset):
    """
    Dataset for satellite imagery and segmentation masks
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir: Directory containing satellite images
            mask_dir: Directory containing segmentation masks
            transform: Augmentation transforms
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Get all image files
        self.images = sorted(list(self.image_dir.glob("*.jpg")) +
                             list(self.image_dir.glob("*.png")) +
                             list(self.image_dir.glob("*.tif")))

        print(f"📁 Found {len(self.images)} images in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = self.mask_dir / img_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create dummy mask if not exists
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize mask if dimensions differ
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()


def get_transforms(train=True, img_size=256):
    """
    Get augmentation transforms for training/validation
    """
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                               rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ], is_check_shapes=False)


def create_dummy_data(output_dir='../data/dummy', n_samples=10, img_size=256):
    """
    Create dummy satellite images and masks for testing
    """
    import os

    images_dir = Path(output_dir) / 'images'
    masks_dir = Path(output_dir) / 'masks'

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎨 Creating {n_samples} dummy samples...")

    for i in range(n_samples):
        # Create synthetic satellite image (greenish for forest)
        image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        image[:, :, 1] = np.clip(image[:, :, 1] * 1.2, 0, 255)  # More green

        # Create synthetic mask with deforestation areas
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Add some deforestation zones (class 1)
        num_zones = np.random.randint(2, 5)
        for _ in range(num_zones):
            x, y = np.random.randint(0, img_size - 50, 2)
            w, h = np.random.randint(20, 50, 2)
            mask[y:y + h, x:x + w] = 1

        # Save
        cv2.imwrite(str(images_dir / f'sat_image_{i:03d}.png'), image)
        cv2.imwrite(str(masks_dir / f'sat_image_{i:03d}.png'), mask)

    print(f"✅ Created {n_samples} dummy samples at {output_dir}")
    return str(images_dir), str(masks_dir)


def test_dataset():
    """Test the dataset loader"""
    print("🧪 Testing Dataset Loader...")

    # Create dummy data
    img_dir, mask_dir = create_dummy_data(n_samples=5)

    # Create dataset
    transforms = get_transforms(train=True, img_size=256)
    dataset = SatelliteDataset(img_dir, mask_dir, transform=transforms)

    print(f"📊 Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Test loading a batch
    for images, masks in dataloader:
        print(f"✅ Batch images shape: {images.shape}")
        print(f"✅ Batch masks shape: {masks.shape}")
        print(f"✅ Image dtype: {images.dtype}, range: [{images.min():.2f}, {images.max():.2f}]")
        print(f"✅ Mask dtype: {masks.dtype}, unique values: {torch.unique(masks)}")
        break

    print("✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
