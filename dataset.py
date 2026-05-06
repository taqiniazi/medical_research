import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class MARTrainDataset(Dataset):
    """Dataset for Metal Artifact Reduction Training"""
    
    def __init__(self, data_path, patch_size=128, batch_num=1000, mask=None):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to training data directory
            patch_size: Size of training patches
            batch_num: Number of batches
            mask: Optional mask for training regions
        """
        self.data_path = data_path
        self.patch_size = patch_size
        self.batch_num = batch_num
        self.mask = mask
        
        # Load training data
        self.load_data()
        
    def load_data(self):
        """Load training data from directory"""
        try:
            # Load metal-corrupted images (Y)
            self.Y_images = np.load(os.path.join(self.data_path, 'train_Y.npy'))
            
            # Load ground truth images (Xgt)
            self.Xgt_images = np.load(os.path.join(self.data_path, 'train_Xgt.npy'))
            
            # Load linear interpolation results (XLI)
            self.XLI_images = np.load(os.path.join(self.data_path, 'train_XLI.npy'))
            
            # Load metal masks if available
            if os.path.exists(os.path.join(self.data_path, 'train_mask.npy')):
                self.metal_masks = np.load(os.path.join(self.data_path, 'train_mask.npy'))
            else:
                # Generate default masks if not available
                self.metal_masks = np.ones_like(self.Y_images)
                
            print(f"Loaded training data: {self.Y_images.shape[0]} images")
            
        except FileNotFoundError as e:
            print(f"Warning: Training data not found at {self.data_path}")
            print("Creating synthetic data for demonstration...")
            self.create_synthetic_data()
    
    def create_synthetic_data(self):
        """Create synthetic data for demonstration purposes"""
        num_samples = 100
        img_size = 256
        
        # Create synthetic ground truth images
        self.Xgt_images = np.random.randn(num_samples, 1, img_size, img_size).astype(np.float32)
        
        # Create synthetic metal artifacts
        self.Y_images = self.Xgt_images.copy()
        
        # Add metal artifacts (simulated)
        for i in range(num_samples):
            # Add some high-intensity regions (simulating metal)
            artifact_mask = np.random.rand(img_size, img_size) > 0.95
            self.Y_images[i, 0] += artifact_mask.astype(np.float32) * 2.0
            
            # Add streaking artifacts
            angle = np.random.rand() * np.pi
            x_coords, y_coords = np.meshgrid(np.arange(img_size), np.arange(img_size))
            streak_pattern = np.sin(x_coords * np.cos(angle) + y_coords * np.sin(angle)) * 0.3
            self.Y_images[i, 0] += streak_pattern
        
        # Create linear interpolation results (simplified)
        self.XLI_images = self.Y_images * 0.8 + self.Xgt_images * 0.2
        
        # Create metal masks
        self.metal_masks = (self.Y_images - self.Xgt_images) > 0.5
        self.metal_masks = self.metal_masks.astype(np.float32)
        
        print(f"Created synthetic training data: {num_samples} samples")
    
    def __len__(self):
        """Return the number of training samples"""
        return self.batch_num
    
    def __getitem__(self, idx):
        """
        Get a training sample
        
        Returns:
            Y: Metal-corrupted image [1, H, W]
            Xgt: Ground truth image [1, H, W]
            XLI: Linear interpolation result [1, H, W]
            mask: Metal artifact mask [1, H, W]
        """
        # Randomly select an image
        img_idx = random.randint(0, len(self.Y_images) - 1)
        
        # Extract random patch
        img_size = self.Y_images.shape[2]
        if self.patch_size < img_size:
            # Extract random patch
            start_x = random.randint(0, img_size - self.patch_size)
            start_y = random.randint(0, img_size - self.patch_size)
            
            Y_patch = self.Y_images[img_idx, :, start_x:start_x + self.patch_size, start_y:start_y + self.patch_size]
            Xgt_patch = self.Xgt_images[img_idx, :, start_x:start_x + self.patch_size, start_y:start_y + self.patch_size]
            XLI_patch = self.XLI_images[img_idx, :, start_x:start_x + self.patch_size, start_y:start_y + self.patch_size]
            mask_patch = self.metal_masks[img_idx, :, start_x:start_x + self.patch_size, start_y:start_y + self.patch_size]
        else:
            # Use full image
            Y_patch = self.Y_images[img_idx]
            Xgt_patch = self.Xgt_images[img_idx]
            XLI_patch = self.XLI_images[img_idx]
            mask_patch = self.metal_masks[img_idx]
        
        # Convert to torch tensors
        Y_tensor = torch.from_numpy(Y_patch.copy())
        Xgt_tensor = torch.from_numpy(Xgt_patch.copy())
        XLI_tensor = torch.from_numpy(XLI_patch.copy())
        mask_tensor = torch.from_numpy(mask_patch.copy())
        
        # Apply additional data augmentation if mask is provided
        if self.mask is not None:
            # Apply training region mask
            mask_tensor = mask_tensor * torch.from_numpy(self.mask)
        
        return Y_tensor, Xgt_tensor, XLI_tensor, mask_tensor


class MARTestDataset(Dataset):
    """Dataset for Metal Artifact Reduction Testing/Evaluation"""
    
    def __init__(self, data_path):
        """
        Initialize the test dataset
        
        Args:
            data_path: Path to test data directory
        """
        self.data_path = data_path
        self.load_test_data()
    
    def load_test_data(self):
        """Load test data"""
        try:
            # Load test images
            self.Y_images = np.load(os.path.join(self.data_path, 'test_Y.npy'))
            self.Xgt_images = np.load(os.path.join(self.data_path, 'test_Xgt.npy'))
            self.XLI_images = np.load(os.path.join(self.data_path, 'test_XLI.npy'))
            
            if os.path.exists(os.path.join(self.data_path, 'test_mask.npy')):
                self.metal_masks = np.load(os.path.join(self.data_path, 'test_mask.npy'))
            else:
                self.metal_masks = np.ones_like(self.Y_images)
                
        except FileNotFoundError:
            print("Test data not found, creating synthetic test data...")
            self.create_synthetic_test_data()
    
    def create_synthetic_test_data(self):
        """Create synthetic test data"""
        num_samples = 20
        img_size = 256
        
        self.Xgt_images = np.random.randn(num_samples, 1, img_size, img_size).astype(np.float32)
        self.Y_images = self.Xgt_images + np.random.randn(num_samples, 1, img_size, img_size) * 0.5
        self.XLI_images = self.Y_images * 0.7 + self.Xgt_images * 0.3
        self.metal_masks = np.ones_like(self.Y_images)
    
    def __len__(self):
        return len(self.Y_images)
    
    def __getitem__(self, idx):
        """Get test sample"""
        Y_tensor = torch.from_numpy(self.Y_images[idx])
        Xgt_tensor = torch.from_numpy(self.Xgt_images[idx])
        XLI_tensor = torch.from_numpy(self.XLI_images[idx])
        mask_tensor = torch.from_numpy(self.metal_masks[idx])
        
        return Y_tensor, Xgt_tensor, XLI_tensor, mask_tensor