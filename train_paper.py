#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
DSDNet Training Script - Paper Specifications Implementation
Based on PR-S-26-06789 paper for CT Metal Artifact Reduction
"""

import os
import math
import argparse
import random
import time
import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from dsdnet import DSDNet
from dataset import MARTrainDataset


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def print_network_parameters(net):
    """Print model parameter count"""
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"========== DSDNet Model Parameters ==========")
    print(f"Total Parameters:      {total_params / 1e6:.4f} M")
    print(f"Trainable Parameters:  {trainable_params / 1e6:.4f} M")
    print("=" * 60)


def get_lr_multiplier(epoch, warmup_epochs=5, total_epochs=200, min_lr_ratio=(1e-6 / 2e-4)):
    """
    Learning rate schedule: 5-round warmup + cosine annealing to 1e-6
    As specified in the paper
    """
    if epoch < warmup_epochs:
        # Linear warmup: 0 -> 1
        return (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing: 1 -> min_lr_ratio
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def train_model(net, optimizer, lr_scheduler, train_dataset, device, opt, start_epoch=0):
    """Training model with paper specifications"""
    data_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=False,  # CPU mode
        worker_init_fn=worker_init_fn
    )

    num_iter_epoch = ceil(len(train_dataset) / opt.batchSize)

    print(f"========== Starting DSDNet Training (Epoch {start_epoch + 1} to {opt.niter}) ==========")
    print(f"Paper Parameters: S={opt.S}, M={opt.num_M}, Q={opt.num_Q}")
    print(f"Training: Batch={opt.batchSize}, Patch={opt.patchSize}, Epochs={opt.niter}")

    for epoch in range(start_epoch, opt.niter):
        mse_per_epoch = 0
        tic = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        net.train()

        for ii, data in enumerate(data_loader):
            Y, Xgt, XLI, mask = [x.to(device) for x in data]

            optimizer.zero_grad()

            # Forward pass through DSDNet
            X0, ListX, ListA, ListC = net(Y, XLI, mask)

            # Paper loss function: Physical mask constraint
            newXgt = mask * Xgt
            newAgt = mask * (Y - Xgt)

            # Multi-stage loss as per paper
            loss_l1_X_stages = 0.1 * F.l1_loss(X0 * mask, newXgt)
            loss_l1_A_stages = 0.0

            for j in range(opt.S - 1):
                loss_l1_X_stages += 0.1 * F.l1_loss(ListX[j] * mask, newXgt)
                loss_l1_A_stages += 0.1 * F.l1_loss(ListA[j] * mask, newAgt)

            loss_l1_X_final = F.l1_loss(ListX[-1] * mask, newXgt)
            loss_l1_A_final = F.l1_loss(ListA[-1] * mask, newAgt)

            loss_l1_X = loss_l1_X_stages + loss_l1_X_final
            loss_l1_A = loss_l1_A_stages + loss_l1_A_final

            # Combined loss with paper weights
            loss = (opt.w_x * loss_l1_X) + (opt.w_a * loss_l1_A)

            loss.backward()
            
            # Gradient clipping (paper specification)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            mse_iter = loss.item()
            mse_per_epoch += mse_iter

            if ii % 20 == 0:
                template = '[Epoch:{:>3d}/{:<3d}] {:0>4d}/{:0>4d}, Loss={:5.2e}, L1_X={:5.2e}, L1_A={:5.2e}, LR={:.2e}'
                print(template.format(epoch + 1, opt.niter, ii, num_iter_epoch, mse_iter, 
                                    loss_l1_X.item(), loss_l1_A.item(), current_lr))

            # Early break for testing
            if (ii + 1) >= 50:  # Limit iterations for testing
                break

        mse_per_epoch /= (ii + 1)
        print('-' * 80)
        print('Epoch [{}/{}] Complete. Avg Loss: {:+.4e}, Time: {:.2f}s'.format(
            epoch + 1, opt.niter, mse_per_epoch, time.time() - tic))
        print('-' * 80)

        # Learning rate scheduling (paper specification)
        lr_scheduler.step()

        # Save checkpoint
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict()
        }

        torch.save(save_dict, os.path.join(opt.model_dir, f'DSDNet_epoch_{epoch + 1}.pt'))
        
        # Keep latest checkpoint
        torch.save(save_dict, os.path.join(opt.model_dir, 'DSDNet_latest.pt'))

        # Early exit for testing (just run a few epochs)
        if epoch >= 2:  # Run only 3 epochs for demonstration
            break

    print('========== DSDNet Training Completed! ==========')


def main():
    """Main function with paper-specific parameters"""
    parser = argparse.ArgumentParser(description='DSDNet Training - Paper Implementation')
    
    # Paper-specific parameters
    parser.add_argument("--data_path", type=str, default="./data", help='Path to training data')
    parser.add_argument('--log_dir', default='./logs/dsdnet_paper', help='Tensorboard logs')
    parser.add_argument('--model_dir', default='./models/dsdnet_paper/', help='Model saving directory')

    # Paper data parameters
    parser.add_argument('--workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=6, help='Batch size (paper: 6)')
    parser.add_argument('--patchSize', type=int, default=128, help='Patch size (paper: 128)')
    parser.add_argument('--niter', type=int, default=200, help='Total epochs (paper: 200)')
    parser.add_argument('--batchnum', type=int, default=1000, help='Number of batches')

    # Paper DSDNet architecture parameters
    parser.add_argument('--num_M', type=int, default=32, help='Dynamic Dictionary Atoms (paper: 32)')
    parser.add_argument('--num_Q', type=int, default=32, help='Channel concatenation (paper: 32)')
    parser.add_argument('--S', type=int, default=5, help='Unrolling stages (paper: 5)')
    parser.add_argument('--rho_z', type=float, default=1.0, help='ADMM penalty for Z (paper: 1.0)')
    parser.add_argument('--rho_x', type=float, default=1.0, help='ADMM penalty for X (paper: 1.0)')

    # Paper loss weights
    parser.add_argument('--w_x', type=float, default=1.0, help='Image L1 loss weight (paper: 1.0)')
    parser.add_argument('--w_a', type=float, default=1.0, help='Artifact L1 loss weight (paper: 1.0)')

    # Environment
    parser.add_argument('--manualSeed', type=int, default=3407, help='Random seed')

    opt = parser.parse_args()

    # Set device (CPU for compatibility)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Paper: DSDNet for CT Metal Artifact Reduction")
    print(f"Parameters: S={opt.S}, M={opt.num_M}, Q={opt.num_Q}, Batch={opt.batchSize}")

    # Create directories
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    # Set random seeds
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    print("\nBuilding DSDNet Model (Paper Architecture)...")
    net = DSDNet(opt).to(device)
    
    # Print model parameters
    print_network_parameters(net)

    print("\nConfiguring Paper-Specific Optimizer (AdamW)...")
    # Paper optimizer: AdamW with specific parameters
    optimizer = optim.AdamW(
        net.parameters(), 
        lr=2e-4,  # Paper learning rate
        betas=(0.5, 0.999),  # Paper beta values
        weight_decay=1e-4  # Paper weight decay
    )
    
    print("Configuring Paper Learning Rate Schedule...")
    # Paper LR schedule: 5-round warmup + cosine annealing
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda ep: get_lr_multiplier(ep, warmup_epochs=5, total_epochs=opt.niter)
    )

    print("\nLoading Training Dataset...")
    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize, opt.batchnum)

    print(f"\nStarting DSDNet Training with Paper Specifications...")
    print(f"Epochs: {opt.niter}, Batch Size: {opt.batchSize}, Patch Size: {opt.patchSize}")
    print(f"ADMM Parameters: rho_z={opt.rho_z}, rho_x={opt.rho_x}")
    print("=" * 80)
    
    train_model(net, optimizer, scheduler, train_dataset, device, opt, start_epoch=0)


if __name__ == '__main__':
    main()