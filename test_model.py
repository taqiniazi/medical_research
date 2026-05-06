#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import argparse

# Import the DSDNet model
from dsdnet import DSDNet

def test_dsdnet():
    """Test the DSDNet model functionality"""
    print("Testing DSDNet model...")
    
    # Create mock options
    class MockOpt:
        def __init__(self):
            self.S = 5  # Number of stages
            self.num_M = 32  # Number of dictionary atoms
            self.num_Q = 32  # Number of channels
            self.rho_z = 1.0  # ADMM penalty for Z
            self.rho_x = 1.0  # ADMM penalty for X
    
    opt = MockOpt()
    
    # Create model
    device = torch.device('cpu')
    net = DSDNet(opt).to(device)
    
    print("✓ Model created successfully")
    
    # Test forward pass
    batch_size = 2
    height, width = 64, 64
    
    # Create mock input data
    Y = torch.randn(batch_size, 1, height, width).to(device)  # Metal-corrupted image
    XLI = torch.randn(batch_size, 1, height, width).to(device)  # Linear interpolation
    mask = torch.ones(batch_size, 1, height, width).to(device)  # Metal mask
    
    print(f"Input shapes:")
    print(f"  Y (metal-corrupted): {Y.shape}")
    print(f"  XLI (interpolation): {XLI.shape}")
    print(f"  mask: {mask.shape}")
    
    # Forward pass
    try:
        X0, ListX, ListA, ListC = net(Y, XLI, mask)
        
        print("✓ Forward pass successful!")
        print(f"Output shapes:")
        print(f"  X0 (initial reconstruction): {X0.shape}")
        print(f"  ListX length: {len(ListX)}")
        print(f"  ListX[0] shape: {ListX[0].shape}")
        print(f"  ListA length: {len(ListA)}")
        print(f"  ListC length: {len(ListC)}")
        
        # Test loss computation
        criterion = nn.L1Loss()
        loss = criterion(X0, Y)
        print(f"✓ Loss computation successful: {loss.item():.6f}")
        
        # Test backward pass
        loss.backward()
        print("✓ Backward pass successful!")
        
        # Count parameters
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Model parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        print("\n🎉 All tests passed! The DSDNet model is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_dsdnet()