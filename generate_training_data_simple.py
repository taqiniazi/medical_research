#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
CT Metal Artifact Training Data Generator (Simple Version)
Creates realistic training data for DSDNet metal artifact reduction
"""

import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path


class SimpleCTMetalArtifactGenerator:
    """Simple generator for CT metal artifact training data"""
    
    def __init__(self, output_dir="./data", image_size=256, num_train=100, num_test=20):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.num_train = num_train
        self.num_test = num_test
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Simple CT Metal Artifact Data Generator")
        print(f"Output directory: {self.output_dir}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Training samples: {num_train}")
        print(f"Test samples: {num_test}")
    
    def generate_synthetic_ct_image(self, size):
        """Generate a synthetic CT image with realistic structures"""
        # Create base image with noise
        image = np.random.normal(0.5, 0.1, size)
        
        # Add circular structures (simulating organs/bones)
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Add concentric circles (simulating bone structures)
        y, x = np.ogrid[:size[0], :size[1]]
        
        # Outer circle (skull/body boundary)
        outer_circle = (x - center_x)**2 + (y - center_y)**2 < (size[0]//2.5)**2
        image[outer_circle] = 0.8
        
        # Inner structures (organs)
        for i in range(3):
            offset_x = np.random.randint(-size[0]//4, size[0]//4)
            offset_y = np.random.randint(-size[1]//4, size[1]//4)
            radius = np.random.randint(size[0]//8, size[0]//6)
            
            circle = (x - center_x - offset_x)**2 + (y - center_y - offset_y)**2 < radius**2
            image[circle] = np.random.uniform(0.3, 0.7)
        
        # Add some linear structures (blood vessels, etc.)
        for i in range(5):
            angle = np.random.uniform(0, 2*np.pi)
            thickness = np.random.randint(2, 5)
            length = np.random.randint(size[0]//4, size[0]//2)
            
            # Create line using numpy operations
            line_mask = np.zeros(size, dtype=bool)
            
            # Line parameters
            start_x, start_y = center_x, center_y
            end_x = start_x + int(length * np.cos(angle))
            end_y = start_y + int(length * np.sin(angle))
            
            # Draw line using Bresenham's algorithm equivalent
            dx = abs(end_x - start_x)
            dy = abs(end_y - start_y)
            x, y = start_x, start_y
            
            # Simple line drawing
            steps = max(dx, dy)
            if steps > 0:
                x_step = (end_x - start_x) / steps
                y_step = (end_y - start_y) / steps
                
                for step in range(int(steps)):
                    x_int = int(x + step * x_step)
                    y_int = int(y + step * y_step)
                    
                    # Draw thick line
                    for dx_thick in range(-thickness//2, thickness//2 + 1):
                        for dy_thick in range(-thickness//2, thickness//2 + 1):
                            x_thick = x_int + dx_thick
                            y_thick = y_int + dy_thick
                            
                            if 0 <= x_thick < size[0] and 0 <= y_thick < size[1]:
                                line_mask[y_thick, x_thick] = True
            
            image[line_mask] = np.random.uniform(0.4, 0.8)
        
        # Add Gaussian blur for smoothness
        image = ndimage.gaussian_filter(image, sigma=1.0)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
        
        return image
    
    def generate_metal_mask(self, size, num_metals=2):
        """Generate metal artifact masks"""
        mask = np.zeros(size, dtype=np.float32)
        
        # Add metal objects (bright regions)
        for i in range(num_metals):
            center_x = np.random.randint(size[0]//4, 3*size[0]//4)
            center_y = np.random.randint(size[1]//4, 3*size[1]//4)
            radius = np.random.randint(5, 15)
            
            y, x = np.ogrid[:size[0], :size[1]]
            circle = (x - center_x)**2 + (y - center_y)**2 < radius**2
            mask[circle] = 1.0
        
        return mask
    
    def create_metal_artifacts(self, clean_image, metal_mask):
        """Create metal artifacts in CT image"""
        # Create corrupted image
        corrupted = clean_image.copy()
        
        # Metal regions become very bright
        corrupted[metal_mask > 0] = 1.0
        
        # Create streaking artifacts (dark and bright bands)
        streaks = np.zeros_like(clean_image)
        
        # Multiple streaking directions
        for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            # Create streak pattern
            y, x = np.ogrid[:clean_image.shape[0], :clean_image.shape[1]]
            
            # Distance from metal objects
            distance_from_metal = ndimage.distance_transform_edt(1 - metal_mask)
            
            # Create sinusoidal streaking pattern
            streak_pattern = np.sin(x * np.cos(angle) + y * np.sin(angle))
            
            # Modulate streaking by distance from metal
            streak_strength = np.exp(-distance_from_metal / 20.0)
            streaks += streak_pattern * streak_strength * 0.2
        
        # Add streaking to corrupted image
        corrupted += streaks
        
        # Add some noise in metal regions
        noise = np.random.normal(0, 0.1, clean_image.shape)
        corrupted += noise * metal_mask
        
        # Clip to valid range
        corrupted = np.clip(corrupted, 0, 1)
        
        return corrupted
    
    def create_linear_interpolation(self, corrupted_image, metal_mask):
        """Create linear interpolation result (simple inpainting)"""
        # Simple linear interpolation by Gaussian smoothing
        interpolated = corrupted_image.copy()
        
        # Smooth the image
        smoothed = ndimage.gaussian_filter(corrupted_image, sigma=3.0)
        
        # Replace metal regions with smoothed values
        interpolated[metal_mask > 0] = smoothed[metal_mask > 0]
        
        return interpolated
    
    def create_training_sample(self, size):
        """Create a complete training sample"""
        # Generate clean CT image
        clean_image = self.generate_synthetic_ct_image(size)
        
        # Generate metal mask
        metal_mask = self.generate_metal_mask(size)
        
        # Create corrupted image with artifacts
        corrupted_image = self.create_metal_artifacts(clean_image, metal_mask)
        
        # Create linear interpolation result
        interpolated_image = self.create_linear_interpolation(corrupted_image, metal_mask)
        
        return {
            'clean': clean_image,
            'corrupted': corrupted_image,
            'interpolated': interpolated_image,
            'metal_mask': metal_mask
        }
    
    def save_dataset(self, samples, prefix):
        """Save dataset to files"""
        # Convert to numpy arrays
        clean_images = np.array([s['clean'] for s in samples])
        corrupted_images = np.array([s['corrupted'] for s in samples])
        interpolated_images = np.array([s['interpolated'] for s in samples])
        metal_masks = np.array([s['metal_mask'] for s in samples])
        
        # Add channel dimension (for PyTorch)
        clean_images = clean_images[:, np.newaxis, :, :]
        corrupted_images = corrupted_images[:, np.newaxis, :, :]
        interpolated_images = interpolated_images[:, np.newaxis, :, :]
        metal_masks = metal_masks[:, np.newaxis, :, :]
        
        # Save as float32
        clean_images = clean_images.astype(np.float32)
        corrupted_images = corrupted_images.astype(np.float32)
        interpolated_images = interpolated_images.astype(np.float32)
        metal_masks = metal_masks.astype(np.float32)
        
        # Save files
        np.save(self.output_dir / f'{prefix}_Xgt.npy', clean_images)
        np.save(self.output_dir / f'{prefix}_Y.npy', corrupted_images)
        np.save(self.output_dir / f'{prefix}_XLI.npy', interpolated_images)
        np.save(self.output_dir / f'{prefix}_mask.npy', metal_masks)
        
        print(f"Saved {prefix} dataset:")
        print(f"  Clean images: {clean_images.shape}")
        print(f"  Corrupted images: {corrupted_images.shape}")
        print(f"  Interpolated images: {interpolated_images.shape}")
        print(f"  Metal masks: {metal_masks.shape}")
    
    def visualize_sample(self, sample, save_path=None):
        """Visualize a training sample"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        axes[0, 0].imshow(sample['clean'], cmap='gray')
        axes[0, 0].set_title('Clean CT Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(sample['corrupted'], cmap='gray')
        axes[0, 1].set_title('Corrupted (with Metal Artifacts)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(sample['interpolated'], cmap='gray')
        axes[1, 0].set_title('Linear Interpolation')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(sample['metal_mask'], cmap='gray')
        axes[1, 1].set_title('Metal Mask')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_data(self):
        """Generate complete training and test datasets"""
        print("\n" + "="*60)
        print("GENERATING TRAINING DATA")
        print("="*60)
        
        # Generate training data
        train_samples = []
        for i in range(self.num_train):
            if i % 20 == 0:
                print(f"Generating training sample {i}/{self.num_train}")
            
            sample = self.create_training_sample((self.image_size, self.image_size))
            train_samples.append(sample)
        
        # Generate test data
        print("\nGENERATING TEST DATA")
        print("="*60)
        
        test_samples = []
        for i in range(self.num_test):
            if i % 5 == 0:
                print(f"Generating test sample {i}/{self.num_test}")
            
            sample = self.create_training_sample((self.image_size, self.image_size))
            test_samples.append(sample)
        
        # Save datasets
        print("\nSAVING DATASETS")
        print("="*60)
        
        self.save_dataset(train_samples, 'train')
        self.save_dataset(test_samples, 'test')
        
        # Save visualization of first training sample
        print("\nSAVING VISUALIZATION")
        print("="*60)
        self.visualize_sample(train_samples[0], self.output_dir / 'sample_visualization.png')
        
        print("\n" + "="*60)
        print("DATA GENERATION COMPLETED!")
        print("="*60)
        print(f"Training data saved to: {self.output_dir}/train_*.npy")
        print(f"Test data saved to: {self.output_dir}/test_*.npy")
        print(f"Sample visualization: {self.output_dir}/sample_visualization.png")


def main():
    """Main function to generate training data"""
    print("Simple CT Metal Artifact Training Data Generator")
    print("This will create realistic training data for DSDNet")
    
    # Configuration
    output_dir = "./data"
    image_size = 256  # 256x256 images
    num_train = 100   # Training samples
    num_test = 20     # Test samples
    
    # Create generator
    generator = SimpleCTMetalArtifactGenerator(
        output_dir=output_dir,
        image_size=image_size,
        num_train=num_train,
        num_test=num_test
    )
    
    # Generate all data
    generator.generate_all_data()


if __name__ == "__main__":
    main()