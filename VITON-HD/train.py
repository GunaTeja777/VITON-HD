import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from the existing VITON-HD repo files
from networks import GMM, BaseNetwork
from datasets import VITONDataset


class TrainOptions:
    """Configuration class for training options"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='VITON-HD GMM Fine-tuning')
        
        # Dataset parameters
        self.parser.add_argument('--dataset_dir', type=str, 
                                default='/content/drive/MyDrive/content/dataset_final',
                                help='Path to dataset directory')
        self.parser.add_argument('--dataset_mode', type=str, default='train',
                                help='Subdirectory name (e.g., train, test)')
        self.parser.add_argument('--dataset_list', type=str, default='train_pairs.txt',
                                help='Filename of the pairs list')
        
        # Checkpoint parameters
        self.parser.add_argument('--checkpoint_dir', type=str,
                                default='/content/drive/MyDrive/content/checkpoints',
                                help='Directory containing pretrained checkpoints')
        self.parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth',
                                help='GMM checkpoint filename')
        self.parser.add_argument('--save_dir', type=str,
                                default='/content/drive/MyDrive/viton_checkpoints/GMM_finetuned',
                                help='Directory to save fine-tuned checkpoints')
        
        # Model parameters
        self.parser.add_argument('--load_height', type=int, default=1024,
                                help='Image height')
        self.parser.add_argument('--load_width', type=int, default=768,
                                help='Image width')
        self.parser.add_argument('--semantic_nc', type=int, default=13,
                                help='Number of semantic classes')
        self.parser.add_argument('--grid_size', type=int, default=5,
                                help='Grid size for TPS transformation')
        self.parser.add_argument('--init_type', type=str, default='xavier',
                                help='Weight initialization type')
        self.parser.add_argument('--init_variance', type=float, default=0.02,
                                help='Variance for weight initialization')
        
        # Training parameters
        self.parser.add_argument('--num_epochs', type=int, default=10,
                                help='Number of training epochs')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                help='Batch size for training')
        self.parser.add_argument('--lr', type=float, default=0.0001,
                                help='Learning rate')
        self.parser.add_argument('--workers', type=int, default=4,
                                help='Number of data loading workers')
        self.parser.add_argument('--shuffle', action='store_true', default=True,
                                help='Shuffle training data')
        
        # Loss weights
        self.parser.add_argument('--lambda_l1', type=float, default=1.0,
                                help='Weight for L1 loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.1,
                                help='Weight for perceptual loss')
        
        # Logging
        self.parser.add_argument('--print_freq', type=int, default=10,
                                help='Frequency of printing training status')
        self.parser.add_argument('--save_freq', type=int, default=1,
                                help='Frequency of saving checkpoints (epochs)')
        
        # GPU
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                help='GPU ids: e.g. 0  0,1,2')
        
    def parse(self):
        self.opt = self.parser.parse_args()
        
        # Set GPU
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        
        return self.opt


class VGGLoss(nn.Module):
    """Perceptual loss using VGG19 features"""
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()
        
        # Use pre-trained VGG19
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.vgg_layers = vgg.features[:35].eval()
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return self.criterion(x_vgg, y_vgg)


class GMMTrainer:
    """Trainer class for GMM fine-tuning"""
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create save directory
        Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.build_model()
        
        # Initialize dataset
        self.build_dataset()
        
        # Initialize optimizer and losses
        self.build_optimizer()
        
    def build_model(self):
        """Initialize GMM model"""
        print("Initializing GMM model...")
        
        # Input channels: cloth (3) + cloth_mask (1) = 4 for inputA
        # Input channels: agnostic (3) + pose (3) + parse (13) = 19 for inputB (approximately)
        # Actual implementation may vary, adjust based on your needs
        inputA_nc = 4  # cloth + cloth_mask
        inputB_nc = 3  # We'll use img_agnostic as inputB
        
        self.model = GMM(self.opt, inputA_nc, inputB_nc)
        
        # Load pretrained weights
        checkpoint_path = os.path.join(self.opt.checkpoint_dir, self.opt.gmm_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained GMM from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            print("Pretrained weights loaded successfully!")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Training from scratch...")
        
        self.model = self.model.to(self.device)
        
    def build_dataset(self):
        """Initialize dataset and dataloader"""
        print("Initializing dataset...")
        
        self.dataset = VITONDataset(self.opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=self.opt.workers,
            pin_memory=True,
            drop_last=True
        )
        
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Number of batches: {len(self.dataloader)}")
        
    def build_optimizer(self):
        """Initialize optimizer and loss functions"""
        print("Initializing optimizer and losses...")
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.opt.lr,
            betas=(0.5, 0.999)
        )
        
        self.criterion_l1 = nn.L1Loss()
        
        # Initialize perceptual loss if weight > 0
        if self.opt.lambda_vgg > 0:
            self.criterion_vgg = VGGLoss().to(self.device)
        else:
            self.criterion_vgg = None
            
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_l1_loss = 0
        epoch_vgg_loss = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.opt.num_epochs}")
        
        for i, batch in enumerate(pbar):
            # Move data to device
            cloth = batch['cloth']['unpaired'].to(self.device)
            cloth_mask = batch['cloth_mask']['unpaired'].to(self.device)
            img_agnostic = batch['img_agnostic'].to(self.device)
            img = batch['img'].to(self.device)
            
            # Prepare inputs
            inputA = torch.cat([cloth, cloth_mask], 1)  # Concatenate cloth and mask
            inputB = img_agnostic
            
            # Forward pass
            self.optimizer.zero_grad()
            theta, warped_grid = self.model(inputA, inputB)
            
            # Warp cloth using the predicted grid
            warped_cloth = F.grid_sample(cloth, warped_grid, padding_mode='border', align_corners=True)
            
            # Compute losses
            loss_l1 = self.criterion_l1(warped_cloth, img)
            loss = self.opt.lambda_l1 * loss_l1
            
            if self.criterion_vgg is not None:
                loss_vgg = self.criterion_vgg(warped_cloth, img)
                loss = loss + self.opt.lambda_vgg * loss_vgg
                epoch_vgg_loss += loss_vgg.item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            epoch_l1_loss += loss_l1.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'l1': loss_l1.item()
            })
            
        # Calculate average losses
        num_batches = len(self.dataloader)
        avg_loss = epoch_loss / num_batches
        avg_l1_loss = epoch_l1_loss / num_batches
        avg_vgg_loss = epoch_vgg_loss / num_batches if self.criterion_vgg else 0
        
        return avg_loss, avg_l1_loss, avg_vgg_loss
    
    def save_checkpoint(self, epoch, avg_loss):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.opt.save_dir,
            f'gmm_epoch_{epoch:03d}_loss_{avg_loss:.4f}.pth'
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Also save as latest
        latest_path = os.path.join(self.opt.save_dir, 'gmm_latest.pth')
        torch.save(self.model.state_dict(), latest_path)
        
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print("Starting GMM Fine-tuning")
        print("="*50 + "\n")
        
        best_loss = float('inf')
        
        for epoch in range(1, self.opt.num_epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            avg_loss, avg_l1_loss, avg_vgg_loss = self.train_epoch(epoch)
            
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.opt.num_epochs} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg L1 Loss: {avg_l1_loss:.4f}")
            if self.criterion_vgg:
                print(f"  Avg VGG Loss: {avg_vgg_loss:.4f}")
            print("-" * 50)
            
            # Save checkpoint
            if epoch % self.opt.save_freq == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(self.opt.save_dir, 'gmm_best.pth')
                torch.save(self.model.state_dict(), best_path)
                print(f"Best model saved with loss: {best_loss:.4f}")
        
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Best loss: {best_loss:.4f}")
        print("="*50)


def main():
    """Main function"""
    # Parse arguments
    opt = TrainOptions().parse()
    
    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 50)
    for key, value in vars(opt).items():
        print(f"{key}: {value}")
    print("-" * 50 + "\n")
    
    # Initialize trainer
    trainer = GMMTrainer(opt)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()