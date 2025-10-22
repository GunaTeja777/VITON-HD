import argparse
import os
import time
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Import from the existing VITON-HD repo files
from networks import GMM, SegGenerator, ALIASGenerator

torch.backends.cudnn.benchmark = True


class CustomVITONDataset(Dataset):
    """Custom dataset for your data structure"""
    def __init__(self, opt):
        super(CustomVITONDataset, self).__init__()
        self.opt = opt
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.semantic_nc = opt.semantic_nc
        self.data_path = opt.dataset_dir
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load pairs from pairs.txt
        pairs_file = os.path.join(self.data_path, 'pairs.txt')
        self.img_names = []
        self.c_names = []
        
        with open(pairs_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        self.img_names.append(parts[0])
                        self.c_names.append(parts[1])
        
        print(f"Loaded {len(self.img_names)} pairs from {pairs_file}")
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        c_name = self.c_names[index]
        
        # Load cloth
        c = Image.open(os.path.join(self.data_path, 'clothes', c_name)).convert('RGB')
        c = c.resize((self.load_width, self.load_height), Image.BILINEAR)
        c_tensor = self.transform(c)
        
        # Load cloth mask
        cm = Image.open(os.path.join(self.data_path, 'cloth-mask', c_name)).convert('L')
        cm = cm.resize((self.load_width, self.load_height), Image.NEAREST)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm_tensor = torch.from_numpy(cm_array).unsqueeze(0)
        
        # Load person image
        img = Image.open(os.path.join(self.data_path, 'images', img_name)).convert('RGB')
        img = img.resize((self.load_width, self.load_height), Image.BILINEAR)
        img_tensor = self.transform(img)
        
        # Load agnostic image
        img_agnostic = Image.open(os.path.join(self.data_path, 'agnostic-v3.2', img_name)).convert('RGB')
        img_agnostic = img_agnostic.resize((self.load_width, self.load_height), Image.BILINEAR)
        img_agnostic_tensor = self.transform(img_agnostic)
        
        # Load pose
        pose_name = img_name.replace('.jpg', '_rendered.png').replace('.png', '_rendered.png')
        pose_path = os.path.join(self.data_path, 'openpose-img', pose_name)
        if not os.path.exists(pose_path):
            pose_name = img_name.replace('.jpg', '_rendered.png')
            pose_path = os.path.join(self.data_path, 'openpose-img', pose_name)
        
        pose_rgb = Image.open(pose_path).convert('RGB')
        pose_rgb = pose_rgb.resize((self.load_width, self.load_height), Image.BILINEAR)
        pose_tensor = self.transform(pose_rgb)
        
        # Load parsing/segmentation
        parse_name = img_name.replace('.jpg', '.png')
        parse_path = os.path.join(self.data_path, 'label', parse_name)
        parse = Image.open(parse_path).convert('L')
        parse = parse.resize((self.load_width, self.load_height), Image.NEAREST)
        parse_array = np.array(parse)
        
        # Convert parse to one-hot encoding
        parse_tensor = torch.from_numpy(parse_array).long()
        parse_onehot = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_onehot.scatter_(0, parse_tensor.unsqueeze(0), 1.0)
        
        # Map to semantic_nc classes (simplified mapping)
        labels = {
            0: [0, 10],  # background
            1: [1, 2],   # hair
            2: [4, 13],  # face
            3: [5, 6, 7],  # upper
            4: [9, 12],  # bottom
            5: [14],     # left_arm
            6: [15],     # right_arm
            7: [16],     # left_leg
            8: [17],     # right_leg
            9: [18],     # left_shoe
            10: [19],    # right_shoe
            11: [8],     # socks
            12: [3, 11]  # noise
        }
        
        parse_agnostic = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(labels)):
            for label in labels[i]:
                if label < 20:
                    parse_agnostic[i] += parse_onehot[label]
        
        result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img_tensor,
            'img_agnostic': img_agnostic_tensor,
            'parse_agnostic': parse_agnostic,
            'pose': pose_tensor,
            'cloth': c_tensor,
            'cloth_mask': cm_tensor,
        }
        return result
    
    def __len__(self):
        return len(self.img_names)


class TrainOptions:
    """Configuration class for training options"""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='VITON-HD Unified Training')
        
        # Model selection
        self.parser.add_argument('--model_type', type=str, required=True, 
                                choices=['gmm', 'seg', 'alias'],
                                help='Model type to train: gmm, seg, or alias')
        
        # Dataset parameters
        self.parser.add_argument('--dataset_dir', type=str, 
                                default='/content/drive/MyDrive/viton_dataset',
                                help='Path to dataset directory')
        
        # Checkpoint parameters
        self.parser.add_argument('--checkpoint_dir', type=str,
                                default='/content/drive/MyDrive/viton_checkpoints',
                                help='Directory containing pretrained checkpoints')
        self.parser.add_argument('--checkpoint_name', type=str, default='',
                                help='Specific checkpoint filename (optional)')
        self.parser.add_argument('--save_dir', type=str,
                                default='/content/drive/MyDrive/viton_checkpoints',
                                help='Base directory to save fine-tuned checkpoints')
        
        # Model parameters
        self.parser.add_argument('--load_height', type=int, default=1024,
                                help='Image height')
        self.parser.add_argument('--load_width', type=int, default=768,
                                help='Image width')
        self.parser.add_argument('--semantic_nc', type=int, default=13,
                                help='Number of semantic classes')
        self.parser.add_argument('--grid_size', type=int, default=5,
                                help='Grid size for TPS transformation (GMM)')
        self.parser.add_argument('--ngf', type=int, default=64,
                                help='Number of generator filters (ALIAS)')
        self.parser.add_argument('--num_upsampling_layers', type=str, default='most',
                                choices=['normal', 'more', 'most'],
                                help='Number of upsampling layers (ALIAS)')
        self.parser.add_argument('--norm_G', type=str, default='spectralaliasinstance',
                                help='Normalization type for generator (ALIAS)')
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
        self.parser.add_argument('--lambda_seg', type=float, default=1.0,
                                help='Weight for segmentation loss (SEG only)')
        
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
        
        # Set default checkpoint names if not provided
        if not self.opt.checkpoint_name:
            checkpoint_map = {
                'gmm': 'gmm_final.pth',
                'seg': 'seg_final.pth',
                'alias': 'alias_final.pth'
            }
            self.opt.checkpoint_name = checkpoint_map[self.opt.model_type]
        
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
        with torch.no_grad():
            x_vgg = self.vgg_layers(x)
            y_vgg = self.vgg_layers(y)
        return self.criterion(x_vgg, y_vgg)


class Trainer:
    """Unified trainer class for GMM, SEG, and ALIAS"""
    def __init__(self, opt):
        self.opt = opt
        self.model_type = opt.model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and len(opt.gpu_ids) > 0 else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Training model: {self.model_type.upper()}")
        
        # Create save directory
        self.save_dir = os.path.join(opt.save_dir, f'{self.model_type.upper()}_finetuned')
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.build_model()
        
        # Initialize dataset
        self.build_dataset()
        
        # Initialize optimizer and losses
        self.build_optimizer()
        
    def build_model(self):
        """Initialize model based on model_type"""
        print(f"Initializing {self.model_type.upper()} model...")
        
        if self.model_type == 'gmm':
            inputA_nc = 4  # cloth (3) + cloth_mask (1)
            inputB_nc = 6  # img_agnostic (3) + pose (3)
            self.model = GMM(self.opt, inputA_nc, inputB_nc)
            
        elif self.model_type == 'seg':
            input_nc = 7  # img_agnostic (3) + pose (3) + warped_cloth_mask (1)
            output_nc = self.opt.semantic_nc
            self.model = SegGenerator(self.opt, input_nc, output_nc)
            
        elif self.model_type == 'alias':
            input_nc = 9  # img_agnostic (3) + pose (3) + warped_cloth (3)
            self.model = ALIASGenerator(self.opt, input_nc)
        
        # Load pretrained weights
        checkpoint_path = os.path.join(self.opt.checkpoint_dir, self.opt.checkpoint_name)
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained {self.model_type.upper()} from {checkpoint_path}")
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
        
        self.dataset = CustomVITONDataset(self.opt)
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
        
        # L1 Loss (used by all models)
        self.criterion_l1 = nn.L1Loss()
        
        # VGG Perceptual Loss (used by GMM and ALIAS)
        if self.model_type in ['gmm', 'alias'] and self.opt.lambda_vgg > 0:
            self.criterion_vgg = VGGLoss().to(self.device)
        else:
            self.criterion_vgg = None
        
        # Cross Entropy Loss (used by SEG)
        if self.model_type == 'seg':
            self.criterion_seg = nn.CrossEntropyLoss()
        else:
            self.criterion_seg = None
    
    def train_gmm_step(self, batch):
        """Training step for GMM"""
        cloth = batch['cloth'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        img_agnostic = batch['img_agnostic'].to(self.device)
        pose = batch['pose'].to(self.device)
        img = batch['img'].to(self.device)
        
        # Prepare inputs
        inputA = torch.cat([cloth, cloth_mask], 1)
        inputB = torch.cat([img_agnostic, pose], 1)
        
        # Forward pass
        theta, warped_grid = self.model(inputA, inputB)
        
        # Warp cloth using the predicted grid
        warped_cloth = F.grid_sample(cloth, warped_grid, mode='bilinear', 
                                     padding_mode='border', align_corners=True)
        
        # Compute losses
        loss_l1 = self.criterion_l1(warped_cloth, img)
        loss = self.opt.lambda_l1 * loss_l1
        
        loss_dict = {'l1': loss_l1.item()}
        
        if self.criterion_vgg is not None:
            loss_vgg = self.criterion_vgg(warped_cloth, img)
            loss = loss + self.opt.lambda_vgg * loss_vgg
            loss_dict['vgg'] = loss_vgg.item()
        
        return loss, loss_dict
    
    def train_seg_step(self, batch):
        """Training step for SEG"""
        img_agnostic = batch['img_agnostic'].to(self.device)
        pose = batch['pose'].to(self.device)
        parse_agnostic = batch['parse_agnostic'].to(self.device)
        cloth_mask = batch['cloth_mask'].to(self.device)
        
        # Prepare input: img_agnostic (3) + pose (3) + cloth_mask (1) = 7 channels
        input_seg = torch.cat([img_agnostic, pose, cloth_mask], 1)
        
        # Forward pass
        pred_seg = self.model(input_seg)
        
        # Ground truth segmentation (convert one-hot to class indices)
        target_seg = torch.argmax(parse_agnostic, dim=1)
        
        # Compute segmentation loss
        loss_seg = self.criterion_seg(pred_seg, target_seg)
        loss = self.opt.lambda_seg * loss_seg
        
        loss_dict = {'seg': loss_seg.item()}
        
        return loss, loss_dict
    
    def train_alias_step(self, batch):
        """Training step for ALIAS"""
        img_agnostic = batch['img_agnostic'].to(self.device)
        pose = batch['pose'].to(self.device)
        img = batch['img'].to(self.device)
        parse_agnostic = batch['parse_agnostic'].to(self.device)
        cloth = batch['cloth'].to(self.device)
        
        # Prepare input: img_agnostic (3) + pose (3) + cloth (3) = 9 channels
        input_alias = torch.cat([img_agnostic, pose, cloth], 1)
        
        # Prepare segmentation inputs
        seg = parse_agnostic
        seg_div = parse_agnostic
        
        # Create misalignment mask
        misalign_mask = torch.ones_like(img_agnostic)[:, :1, :, :]
        
        # Forward pass
        pred_img = self.model(input_alias, seg, seg_div, misalign_mask)
        
        # Compute losses
        loss_l1 = self.criterion_l1(pred_img, img)
        loss = self.opt.lambda_l1 * loss_l1
        
        loss_dict = {'l1': loss_l1.item()}
        
        if self.criterion_vgg is not None:
            loss_vgg = self.criterion_vgg(pred_img, img)
            loss = loss + self.opt.lambda_vgg * loss_vgg
            loss_dict['vgg'] = loss_vgg.item()
        
        return loss, loss_dict
            
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {}
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.opt.num_epochs}")
        
        for i, batch in enumerate(pbar):
            # Forward pass based on model type
            self.optimizer.zero_grad()
            
            if self.model_type == 'gmm':
                loss, loss_dict = self.train_gmm_step(batch)
            elif self.model_type == 'seg':
                loss, loss_dict = self.train_seg_step(batch)
            elif self.model_type == 'alias':
                loss, loss_dict = self.train_alias_step(batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            # Update statistics
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0
                epoch_losses[key] += value
            
            # Update progress bar
            pbar.set_postfix(loss_dict)
        
        # Calculate average losses
        num_batches = len(self.dataloader)
        avg_losses = {key: value / num_batches for key, value in epoch_losses.items()}
        
        return avg_losses
    
    def save_checkpoint(self, epoch, avg_losses):
        """Save model checkpoint"""
        total_loss = sum(avg_losses.values())
        checkpoint_path = os.path.join(
            self.save_dir,
            f'{self.model_type}_epoch_{epoch:03d}_loss_{total_loss:.4f}.pth'
        )
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Also save as latest
        latest_path = os.path.join(self.save_dir, f'{self.model_type}_latest.pth')
        torch.save(self.model.state_dict(), latest_path)
        
    def train(self):
        """Main training loop"""
        print("\n" + "="*50)
        print(f"Starting {self.model_type.upper()} Fine-tuning")
        print("="*50 + "\n")
        
        best_loss = float('inf')
        
        for epoch in range(1, self.opt.num_epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            avg_losses = self.train_epoch(epoch)
            
            epoch_time = time.time() - start_time
            total_loss = sum(avg_losses.values())
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.opt.num_epochs} Summary:")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Total Loss: {total_loss:.4f}")
            for key, value in avg_losses.items():
                print(f"  {key.upper()} Loss: {value:.4f}")
            print("-" * 50)
            
            # Save checkpoint
            if epoch % self.opt.save_freq == 0:
                self.save_checkpoint(epoch, avg_losses)
            
            # Save best model
            if total_loss < best_loss:
                best_loss = total_loss
                best_path = os.path.join(self.save_dir, f'{self.model_type}_best.pth')
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
    trainer = Trainer(opt)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
    
    
    
