#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Simplified DSDNet Training Script for CPU Testing
This is a simplified version that works without CUDA
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


class SimpleLoss(nn.Module):
    """Simple L1 loss for CPU training"""
    def __init__(self):
        super(SimpleLoss, self).__init__()
    
    def forward(self, x, y):
        return F.l1_loss(x, y)


def print_network_parameters(net):
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"========== Model Parameters ==========")
    print(f"Total Parameters:      {total_params / 1e6:.4f} M")
    print(f"Trainable Parameters:  {trainable_params / 1e6:.4f} M")
    print("=" * 60)


def train_model(net, optimizer, lr_scheduler, train_dataset, device, opt, start_epoch=0):
    data_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=False,  # CPU mode
        worker_init_fn=worker_init_fn
    )

    num_iter_epoch = ceil(len(train_dataset) / opt.batchSize)
    
    # Simple loss function for CPU
    loss_fn = SimpleLoss()

    print(f"========== Start training from epoch {start_epoch + 1} to {opt.niter} ==========")

    for epoch in range(start_epoch, opt.niter):
        mse_per_epoch = 0
        tic = time.time()

        current_lr = optimizer.param_groups[0]['lr']
        net.train()

        for ii, data in enumerate(data_loader):
            Y, Xgt, XLI, mask = [x.to(device) for x in data]

            optimizer.zero_grad()

            X0, ListX, ListA, _ = net(Y, XLI, mask)

            # Compute losses
            newXgt = mask * Xgt
            
            # Simple L1 loss for image reconstruction
            loss_l1_X = F.l1_loss(ListX[-1] * mask, newXgt)
            loss_l1_A = F.l1_loss(ListA[-1] * mask, mask * (Y - Xgt))
            
            # Combined loss
            loss = loss_l1_X + loss_l1_A

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            mse_iter = loss.item()
            mse_per_epoch += mse_iter

            if ii % 10 == 0:
                template = '[Epoch:{:>3d}/{:<3d}] {:0>4d}/{:0>4d}, Loss={:5.2e}, LR={:.2e}'
                print(template.format(epoch + 1, opt.niter, ii, num_iter_epoch, mse_iter, current_lr))

            if (ii + 1) % 50 == 0:
                break  # Limit iterations for testing

        mse_per_epoch /= (ii + 1)
        print('-' * 60)
        print('Epoch [{}/{}] Done. Avg Loss: {:+.4e}, Time: {:.2f}s'.format(epoch + 1, opt.niter, mse_per_epoch,
                                                                            time.time() - tic))
        print('-' * 60)

        # Learning rate scheduling
        lr_scheduler.step()

        # Save checkpoint
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict()
        }

        torch.save(save_dict, os.path.join(opt.model_dir, 'DSDNet_latest.pt'))
        
        # Early exit for testing
        if epoch >= 0:  # Just run 1 epoch for testing
            break

    print('========== Training completed! ==========')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data", help='path to training data')
    parser.add_argument('--log_dir', default='./logs/dsdnet_test', help='tensorboard logs')
    parser.add_argument('--model_dir', default='./models/dsdnet_test/', help='saving model')

    # Training parameters
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
    parser.add_argument('--patchSize', type=int, default=64, help='the height/width of the input image')
    parser.add_argument('--niter', type=int, default=1, help='total number of training epochs')
    parser.add_argument('--batchnum', type=int, default=10, help='the number of batch')

    # DSDNet parameters
    parser.add_argument('--num_M', type=int, default=16, help='the number of Dynamic Dictionary Atoms')
    parser.add_argument('--num_Q', type=int, default=16, help='the number of channel concatenation')
    parser.add_argument('--S', type=int, default=3, help='Stage number (Unrolling iterations)')
    parser.add_argument('--rho_z', type=float, default=1.0, help='ADMM penalty for Z')
    parser.add_argument('--rho_x', type=float, default=1.0, help='ADMM penalty for X')

    # Environment
    parser.add_argument('--manualSeed', type=int, default=3407, help='manual seed')

    opt = parser.parse_args()

    # Set device (CPU for testing)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    # Set random seeds
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    print("Building DSDNet Model...")
    net = DSDNet(opt).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create optimizer
    optimizer = optim.AdamW(net.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=1e-6)

    print("Loading datasets...")
    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize, opt.batchnum)

    print("Starting training...")
    train_model(net, optimizer, scheduler, train_dataset, device, opt, start_epoch=0)


if __name__ == '__main__':
    main()