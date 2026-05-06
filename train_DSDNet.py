#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import math  # 用于计算余弦退火

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import random
import time
import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision.models as models

# 假设这些模块与 main.py 同级目录
from dsdnet import DSDNet
from dataset import MARTrainDataset


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            self.vgg_layers = nn.Sequential(*list(vgg.children())[:9]).to(device)
            for param in self.vgg_layers.parameters():
                param.requires_grad = False
            self.vgg_layers.eval()
            self.available = True
        except:
            print("Warning: VGG16 weights not available, using L1 loss instead")
            self.available = False

    def forward(self, x, y):
        if self.available:
            x_vgg = self.vgg_layers(x.repeat(1, 3, 1, 1))
            y_vgg = self.vgg_layers(y.repeat(1, 3, 1, 1))
            return F.mse_loss(x_vgg, y_vgg)
        else:
            # Fallback to L1 loss if VGG is not available
            return F.l1_loss(x, y)


# ==================================================================
# 打印模型参数量功能
# ==================================================================
def print_network_parameters(net):
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"========== Model Parameters ==========")
    print(f"Total Parameters:      {total_params / 1e6:.4f} M")
    print(f"Trainable Parameters:  {trainable_params / 1e6:.4f} M")
    print("=" * 60)


# ==================================================================
# 严格按照论文编写：5 轮 Warmup + Cosine Annealing 到 1e-6
# ==================================================================
def get_lr_multiplier(epoch, warmup_epochs=5, total_epochs=200, min_lr_ratio=(1e-6 / 2e-4)):
    if epoch < warmup_epochs:
        # 线性预热：0 -> 1
        return (epoch + 1) / warmup_epochs
    else:
        # 余弦退火：1 -> min_lr_ratio
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def train_model(net, optimizer, lr_scheduler, train_dataset, device, opt, start_epoch=0):
    data_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=opt.use_gpu,  # Only use pin_memory if using GPU
        worker_init_fn=worker_init_fn
    )

    num_iter_epoch = ceil(len(train_dataset) / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)

    step = start_epoch * num_iter_epoch

    perceptual_loss_fn = PerceptualLoss(device)

    print(f"========== Start training from epoch {start_epoch + 1} to {opt.niter} ==========")

    # 【修改】：循环从 start_epoch 开始
    for epoch in range(start_epoch, opt.niter):
        mse_per_epoch = 0
        tic = time.time()

        # 获取当前 Epoch 学习率
        current_lr = optimizer.param_groups[0]['lr']
        net.train()

        for ii, data in enumerate(data_loader):
            Y, Xgt, XLI, mask = [x.to(device) for x in data]

            optimizer.zero_grad()

            X0, ListX, ListA, _ = net(Y, XLI, mask)

            # 物理掩膜：只在正常的组织区域计算 Loss
            newXgt = mask * Xgt
            newAgt = mask * (Y - Xgt)

            loss_l1_X_stages = 0.1 * F.l1_loss(X0 * mask, newXgt)
            loss_l1_A_stages = 0.0

            for j in range(opt.S - 1):
                loss_l1_X_stages += 0.1 * F.l1_loss(ListX[j] * mask, newXgt)
                loss_l1_A_stages += 0.1 * F.l1_loss(ListA[j] * mask, newAgt)

            loss_l1_X_final = F.l1_loss(ListX[-1] * mask, newXgt)
            loss_l1_A_final = F.l1_loss(ListA[-1] * mask, newAgt)

            loss_l1_X = loss_l1_X_stages + loss_l1_X_final
            loss_l1_A = loss_l1_A_stages + loss_l1_A_final

            loss_perc = perceptual_loss_fn(ListX[-1] * mask, newXgt)

            loss = (opt.w_x * loss_l1_X) + (opt.w_a * loss_l1_A) + (opt.w_p * loss_perc)

            loss.backward()

            # 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            mse_iter = loss.item()
            mse_per_epoch += mse_iter

            if ii % 50 == 0:
                template = '[Epoch:{:>3d}/{:<3d}] {:0>4d}/{:0>4d}, Total={:5.2e}, L1_X={:5.2e}, L1_A={:5.2e}, Perc={:5.2e}, LR={:.2e}'
                print(template.format(epoch + 1, opt.niter, ii, num_iter_epoch, mse_iter, loss_l1_X.item(),
                                      loss_l1_A.item(), loss_perc.item(), current_lr))

            if (step + 1) % 10 == 0:
                writer.add_scalar('Train/Loss_Total', mse_iter, step)
                writer.add_scalar('Train/Loss_L1_X_Final', loss_l1_X_final.item(), step)
                writer.add_scalar('Train/Loss_L1_A_Final', loss_l1_A_final.item(), step)
                writer.add_scalar('Train/Loss_Perc', loss_perc.item(), step)
                writer.add_scalar('Train/Learning_Rate', current_lr, step)
            step += 1

        mse_per_epoch /= (ii + 1)
        print('-' * 90)
        print('Epoch [{}/{}] Done. Avg Loss: {:+.4e}, Time: {:.2f}s'.format(epoch + 1, opt.niter, mse_per_epoch,
                                                                            time.time() - tic))
        print('-' * 90)

        # 调度器推进
        lr_scheduler.step()

        # 【核心修改】：统一保存为字典格式，包含优化器和调度器状态，供断点续训使用
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict()
        }

        # 保存最新模型
        torch.save(save_dict, os.path.join(opt.model_dir, 'DSDNet_latest.pt'))

        # 定期保存 Checkpoint
        if (epoch + 1) % 10 == 0 or epoch == opt.niter - 1:
            torch.save(save_dict, os.path.join(opt.model_dir, f'DSDNet_checkpoint_{epoch + 1}.pt'))

    writer.close()
    print('========== Reach the maximal epochs! Finish training ==========')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="D:/DICDNet-main/data/train/", help='path to training data')
    parser.add_argument('--log_dir', default='./logs/dsdnet_official_f修改动态4_S=5', help='tensorboard logs')
    parser.add_argument('--model_dir', default='./models/dsdnet_official_修改动态4_S=5/', help='saving model')

    # 【新增功能：断点续训参数】
    parser.add_argument('--resume', type=bool, default=True, help='If true, automatically resume from DSDNet_latest.pt')
    parser.add_argument('--checkpoint_path', type=str, default=80,
                        help='Path to a specific checkpoint to resume from')

    # 数据参数 (严格遵循论文)
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
    parser.add_argument('--patchSize', type=int, default=128, help='the height/width of the input image')
    parser.add_argument('--niter', type=int, default=200, help='total number of training epochs')
    parser.add_argument('--batchnum', type=int, default=1000, help='the number of batch')

    # DSDNet 网络物理参数
    parser.add_argument('--num_M', type=int, default=32, help='the number of Dynamic Dictionary Atoms')
    parser.add_argument('--num_Q', type=int, default=32, help='the number of channel concatenation')
    parser.add_argument('--S', type=int, default=5, help='Stage number (Unrolling iterations)')
    parser.add_argument('--rho_z', type=float, default=1.0, help='ADMM penalty for Z')
    parser.add_argument('--rho_x', type=float, default=1.0, help='ADMM penalty for X')

    # 损失函数权重
    parser.add_argument('--w_x', type=float, default=1.0, help='Weight for Image L1 Loss')
    parser.add_argument('--w_a', type=float, default=1.0, help='Weight for Artifact L1 Loss')
    parser.add_argument('--w_p', type=float, default=0.002, help='Weight for Perceptual Loss (Must be small)')

    # 运行环境
    parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument('--manualSeed', type=int, default=3407, help='manual seed')

    opt = parser.parse_args()

    if opt.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.model_dir, exist_ok=True)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    print("Building DSDNet Model...")
    net = DSDNet(opt).to(device)

    # 【新增】：调用打印参数量函数
    print_network_parameters(net)

    # 【核心：严格按照论文配置 AdamW 优化器】
    optimizer = optim.AdamW(net.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)

    # 【核心：定制包含 Warm-up 的余弦退火调度器】
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: get_lr_multiplier(ep, warmup_epochs=5,
                                                                                              total_epochs=opt.niter))

    # ==================================================================
    # 断点续训 (Resume) 逻辑
    # ==================================================================
    start_epoch = 0
    resume_file = None

    # 优先检查是否指定了明确的路径，其次检查是否开启了自动寻找 latest
    if opt.checkpoint_path and os.path.exists(opt.checkpoint_path):
        resume_file = opt.checkpoint_path
    elif opt.resume:
        latest_path = os.path.join(opt.model_dir, 'DSDNet_latest.pt')
        if os.path.exists(latest_path):
            resume_file = latest_path

    if resume_file:
        print(f"===> Loading Checkpoint: {resume_file}")
        checkpoint = torch.load(resume_file, map_location=device)

        # 兼容性检查：判断是旧版本只存了 weight，还是新版本存了 dict
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"===> Resumed successfully! Training will start from epoch {start_epoch + 1}.")
        else:
            net.load_state_dict(checkpoint)
            print("===> Note: Loaded model weights only (old format). Epoch and Optimizer start from 0.")
    else:
        print("===> No checkpoint found or resume not requested. Training from scratch.")
    # ==================================================================

    print("Loading datasets...")
    try:
        train_mask = np.load(os.path.join(opt.data_path, 'trainmask.npy'))
    except FileNotFoundError:
        train_mask = None
        print("[!] trainmask.npy not found.")

    train_dataset = MARTrainDataset(opt.data_path, opt.patchSize, int(opt.batchSize * opt.batchnum), train_mask)

    # 启动训练，传入 start_epoch
    train_model(net, optimizer, scheduler, train_dataset, device, opt, start_epoch=start_epoch)


if __name__ == '__main__':
    main()