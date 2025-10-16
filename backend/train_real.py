import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.unet import UNet
from utils.dataset import SatelliteDataset, get_transforms


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function with class weights (deforestation is rare, so weight it more)
        # Class 0: Background, 1: Deforestation, 2: Urban, 3: Water
        class_weights = torch.tensor([1.0, 3.0, 2.0, 1.5]).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        print(f"🎯 Training on device: {device}")
        print(f"⚖️  Using class weights: {class_weights.tolist()}")
    
    def calculate_metrics(self, outputs, masks):
        """Calculate accuracy and per-class metrics"""
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == masks).float().sum()
        total = masks.numel()
        accuracy = (correct / total).item() * 100
        
        # Per-class accuracy
        class_acc = {}
        for class_id in range(4):
            class_mask = masks == class_id
            if class_mask.sum() > 0:
                class_correct = ((preds == class_id) & class_mask).float().sum()
                class_acc[class_id] = (class_correct / class_mask.sum()).item() * 100
        
        return accuracy, class_acc
    
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate accuracy
            acc, _ = self.calculate_metrics(outputs, masks)
            
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
        all_class_acc = {0: [], 1: [], 2: [], 3: []}
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                acc, class_acc = self.calculate_metrics(outputs, masks)
                
                running_loss += loss.item()
                running_acc += acc
                
                for class_id, ca in class_acc.items():
                    all_class_acc[class_id].append(ca)
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = running_acc / len(self.val_loader)
        
        # Average class accuracies
        avg_class_acc = {k: sum(v)/len(v) if v else 0 for k, v in all_class_acc.items()}
        
        return val_loss, val_acc, avg_class_acc
    
    def train(self, num_epochs, save_path='models/unet_real_deforestation.pth'):
        """Train the model"""
        print(f"\n{'='*70}")
        print(f"🚀 Starting Training on Real Deforestation Data")
        print(f"{'='*70}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        start_time = time.time()
        
        class_names = {0: "No Change", 1: "Deforestation", 2: "Urban", 3: "Water"}
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"📍 Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, class_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"{'─'*70}")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            print(f"\n   Per-Class Validation Accuracy:")
            for class_id, acc in class_acc.items():
                print(f"      {class_names[class_id]:15s}: {acc:.2f}%")
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'class_acc': class_acc,
                    'history': self.history
                }, save_path)
                print(f"\n   ✅ Best model saved! (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"\n   ⏸️  No improvement. Patience: {patience_counter}/{max_patience}")
                
                if patience_counter >= max_patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"🎉 Training Complete!")
        print(f"{'='*70}")
        print(f"⏱️  Total time: {elapsed_time/60:.2f} minutes")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved to: {save_path}")
        print(f"{'='*70}\n")
        
        # Plot training history
        self.plot_history(save_path.replace('.pth', '_history.png'))
    
    def plot_history(self, save_path):
        """Plot training history"""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Training history plot saved to: {save_path}")


def main():
    """Main training function with real data"""
    print("🌍 VisionEarth - Training on Real Deforestation Data")
    print("="*70)
    
    # Configuration
    IMG_SIZE = 256
    BATCH_SIZE = 4
    NUM_EPOCHS = 50  # More epochs for real data
    LEARNING_RATE = 0.0003
    NUM_CLASSES = 4
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU (training will be slower)")
    
    # Dataset paths
    data_dir = Path('data/real_dataset')
    img_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    if not img_dir.exists():
        print(f"❌ Dataset not found at {img_dir}")
        print("Please run: python3 utils/prepare_real_dataset.py")
        return
    
    # Count samples
    images = list(img_dir.glob('*.png'))
    print(f"\n📊 Dataset Information:")
    print(f"   Total samples: {len(images)}")
    
    # Create dataset
    train_transforms = get_transforms(train=True, img_size=IMG_SIZE)
    
    full_dataset = SatelliteDataset(img_dir, mask_dir, transform=train_transforms)
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_transforms = get_transforms(train=False, img_size=IMG_SIZE)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\n🤖 Creating U-Net model...")
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
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
        save_path='models/unet_real_deforestation.pth'
    )
    
    print("\n✅ Training complete! Model ready for real deforestation detection.")
    print("\n💡 Next steps:")
    print("   1. Restart your API: python3 app.py")
    print("   2. Update predictor to use new model")
    print("   3. Test with real satellite images!")


if __name__ == "__main__":
    main()