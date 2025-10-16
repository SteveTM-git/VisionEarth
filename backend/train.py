import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from utils.dataset import SatelliteDataset, get_transforms, create_dummy_data


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function - CrossEntropyLoss for multi-class segmentation
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        print(f"🎯 Training on device: {device}")
    
    def calculate_accuracy(self, outputs, masks):
        """Calculate pixel-wise accuracy"""
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == masks).float().sum()
        total = masks.numel()
        return (correct / total).item() * 100
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            acc = self.calculate_accuracy(outputs, masks)
            
            # Update metrics
            running_loss += loss.item()
            running_acc += acc
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_acc / len(self.train_loader)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate accuracy
                acc = self.calculate_accuracy(outputs, masks)
                
                running_loss += loss.item()
                running_acc += acc
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = running_acc / len(self.val_loader)
        
        return val_loss, val_acc
    
    def train(self, num_epochs, save_path='models/unet_deforestation.pth'):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\n📍 Epoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, save_path)
                print(f"   ✅ Best model saved! (val_loss: {val_loss:.4f})")
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎉 Training Complete!")
        print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved to: {save_path}")
        print(f"{'='*60}\n")


def main():
    """Main training function"""
    print("🌍 VisionEarth - Deforestation Detection Training")
    print("=" * 60)
    
    # Configuration
    IMG_SIZE = 256
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = 4  # Background, Deforestation, Urban, Water
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU (training will be slower)")
    
    # Create dummy data for demonstration
    print("\n📁 Preparing dataset...")
    img_dir, mask_dir = create_dummy_data(
        output_dir='data/train_dummy',
        n_samples=50,
        img_size=IMG_SIZE
    )
    
    val_img_dir, val_mask_dir = create_dummy_data(
        output_dir='data/val_dummy',
        n_samples=10,
        img_size=IMG_SIZE
    )
    
    # Create datasets
    train_transforms = get_transforms(train=True, img_size=IMG_SIZE)
    val_transforms = get_transforms(train=False, img_size=IMG_SIZE)
    
    train_dataset = SatelliteDataset(img_dir, mask_dir, transform=train_transforms)
    val_dataset = SatelliteDataset(val_img_dir, val_mask_dir, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Mac compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\n🤖 Creating U-Net model...")
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {total_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=LEARNING_RATE
    )
    
    # Train
    trainer.train(
        num_epochs=NUM_EPOCHS,
        save_path='models/unet_deforestation.pth'
    )
    
    print("✅ Training complete! Model ready for inference.")
    print("\n💡 Next steps:")
    print("   1. Test the model with real satellite images")
    print("   2. Create API endpoints for predictions")
    print("   3. Build the React dashboard")


if __name__ == "__main__":
    main()