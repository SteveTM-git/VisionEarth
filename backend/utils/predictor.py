import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeforestationPredictor:
    """Predictor for deforestation detection using trained U-Net model"""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = UNet(n_channels=3, n_classes=4, bilinear=True)

        # Load trained weights
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Model loaded from {model_path}")
        else:
            print(f"⚠️ Model not found at {model_path}, using untrained model")

        self.model.to(self.device)
        self.model.eval()

        # Class names and colors for visualization
        self.class_names = {
            0: "No Change",
            1: "Deforestation",
            2: "Urban Expansion",
            3: "Water Bodies"
        }

        self.class_colors = {
            0: [0, 255, 0],      # Green - No change
            1: [255, 0, 0],      # Red - Deforestation
            2: [128, 128, 128],  # Gray - Urban
            3: [0, 0, 255]       # Blue - Water
        }

        # Preprocessing transform
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

        return image_tensor

    def predict(self, image):
        """
        Predict deforestation mask for an image with enhanced detection

        Args:
            image: numpy array (H, W, 3)

        Returns:
            mask: numpy array (H, W) with class predictions
            probabilities: numpy array (H, W, num_classes) with class probabilities
        """
        original_shape = image.shape[:2]

        # Preprocess
        image_tensor = self.preprocess_image(image).to(self.device)

        # Predict with model
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        # Convert to numpy
        mask = predictions.cpu().numpy()[0]
        probs = probabilities.cpu().numpy()[0].transpose(1, 2, 0)

        # Resize back to original size
        mask = cv2.resize(mask.astype(np.uint8),
                          (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_NEAREST)

        probs = cv2.resize(probs,
                           (original_shape[1], original_shape[0]),
                           interpolation=cv2.INTER_LINEAR)

        # ENHANCEMENT: Add image analysis for better water and urban detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect water bodies (blue hues)
        water_mask = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130) & (hsv[:, :, 1] > 40)) | \
                     ((hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 135) &
                      (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 200))

        # Apply water detection
        if water_mask.sum() > (mask.size * 0.01):  # At least 1% of image
            mask[water_mask] = 3

        # Detect urban/built areas (gray, low saturation)
        urban_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 70) & (hsv[:, :, 2] < 180)

        # Also detect roads/cleared areas (brownish-gray)
        urban_mask2 = (gray > 100) & (gray < 160) & (hsv[:, :, 1] < 60)
        urban_combined = urban_mask | urban_mask2

        # Apply urban detection
        if urban_combined.sum() > (mask.size * 0.01):  # At least 1% of image
            mask[urban_combined] = 2

        # Enhance deforestation detection (brown/bare soil)
        brown_mask = ((hsv[:, :, 0] > 5) & (hsv[:, :, 0] < 25) &
                      (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 50))
        if brown_mask.sum() > (mask.size * 0.005):  # At least 0.5% of image
            mask[brown_mask] = 1

        return mask, probs

    def visualize_prediction(self, image, mask, alpha=0.5):
        """
        Create visualization overlay of predictions on original image

        Args:
            image: Original image (H, W, 3)
            mask: Prediction mask (H, W)
            alpha: Transparency for overlay

        Returns:
            vis_image: Visualization image
        """
        # Create colored mask
        colored_mask = np.zeros_like(image)
        for class_id, color in self.class_colors.items():
            colored_mask[mask == class_id] = color

        # Blend with original image
        vis_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        return vis_image

    def get_statistics(self, mask):
        """
        Calculate statistics from prediction mask

        Returns:
            dict: Statistics for each class
        """
        total_pixels = mask.size
        stats = {}

        for class_id, class_name in self.class_names.items():
            count = np.sum(mask == class_id)
            percentage = (count / total_pixels) * 100
            stats[class_name] = {
                'pixels': int(count),
                'percentage': float(percentage)
            }

        return stats

    def predict_from_path(self, image_path):
        """
        Predict from image file path

        Returns:
            dict: Prediction results
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Predict
        mask, probs = self.predict(image)

        # Get statistics
        stats = self.get_statistics(mask)

        # Create visualization
        vis_image = self.visualize_prediction(image, mask)

        return {
            'mask': mask,
            'probabilities': probs,
            'statistics': stats,
            'visualization': vis_image,
            'original_image': image
        }


def test_predictor():
    """Test the predictor"""
    print("🧪 Testing Deforestation Predictor...")

    # Check if model exists
    model_path = '../models/unet_deforestation.pth'
    if not Path(model_path).exists():
        print(f"⚠️ Model not found at {model_path}")
        print("Run training first: python3 train.py")
        return

    # Create predictor
    predictor = DeforestationPredictor(model_path)

    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    print("📸 Running prediction on test image...")
    mask, probs = predictor.predict(test_image)

    print(f"✅ Prediction shape: {mask.shape}")
    print(f"✅ Unique classes in prediction: {np.unique(mask)}")

    # Get statistics
    stats = predictor.get_statistics(mask)

    print("\n📊 Prediction Statistics:")
    for class_name, data in stats.items():
        print(f"   {class_name}: {data['percentage']:.2f}%")

    # Create visualization
    vis = predictor.visualize_prediction(test_image, mask)
    print(f"✅ Visualization created: {vis.shape}")

    print("\n✅ Predictor test passed!")


if __name__ == "__main__":
    test_predictor()
