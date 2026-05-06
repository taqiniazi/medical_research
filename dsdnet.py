import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DynamicDictionary(nn.Module):
    """Dynamic Dictionary for sparse representation"""
    def __init__(self, num_atoms=32, feature_dim=64):
        super(DynamicDictionary, self).__init__()
        self.num_atoms = num_atoms
        self.feature_dim = feature_dim
        
        # Dictionary atoms
        self.dictionary = nn.Parameter(torch.randn(num_atoms, feature_dim) * 0.1)
        
    def forward(self, x):
        # Compute sparse coefficients
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # Compute coefficients using soft thresholding
        coeffs = torch.matmul(x_flat, self.dictionary.t())  # [B, H*W, num_atoms]
        coeffs = F.softshrink(coeffs, lambd=0.1)  # Sparse coding
        
        # Reconstruct
        reconstruction = torch.matmul(coeffs, self.dictionary)  # [B, H*W, C]
        reconstruction = reconstruction.permute(0, 2, 1).view(B, C, H, W)
        
        return reconstruction, coeffs


class ADMMBlock(nn.Module):
    """ADMM optimization block for unrolling"""
    def __init__(self, num_features=64, num_atoms=32, rho_z=1.0, rho_x=1.0):
        super(ADMMBlock, self).__init__()
        self.rho_z = rho_z
        self.rho_x = rho_x
        
        # Proximal operators
        self.prox_x = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1)
        )
        
        self.prox_z = DynamicDictionary(num_atoms, num_features)
        
        # Lagrange multiplier update
        self.dual_update = nn.Conv2d(num_features, num_features, 1)
        
    def forward(self, Y, XLI, mask, X_prev, Z_prev, U_prev):
        # ADMM step 1: X-update (proximal operator)
        X_update = self.prox_x(Y - XLI + self.rho_z * (Z_prev - U_prev))
        X_new = (1 - mask) * X_prev + mask * X_update
        
        # ADMM step 2: Z-update (sparse coding)
        Z_new, coeffs = self.prox_z(X_new + U_prev)
        
        # ADMM step 3: Dual variable update
        U_new = U_prev + X_new - Z_new
        
        return X_new, Z_new, U_new, coeffs


class DSDNet(nn.Module):
    """Dynamic Sparse Dictionary Network for Metal Artifact Reduction"""
    
    def __init__(self, opt):
        super(DSDNet, self).__init__()
        self.S = opt.S  # Number of stages (unrolling iterations)
        self.num_M = opt.num_M  # Number of dictionary atoms
        self.num_Q = opt.num_Q  # Number of channels
        
        # Initial feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_Q, 3, padding=1)
        )
        
        # ADMM blocks for unrolling
        self.admm_blocks = nn.ModuleList([
            ADMMBlock(self.num_Q, self.num_M, opt.rho_z, opt.rho_x)
            for _ in range(self.S)
        ])
        
        # Reconstruction layers
        self.reconstruction = nn.Sequential(
            nn.Conv2d(self.num_Q, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        
        # Artifact estimation
        self.artifact_estimation = nn.Sequential(
            nn.Conv2d(self.num_Q, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        
    def forward(self, Y, XLI, mask):
        """
        Forward pass of DSDNet
        
        Args:
            Y: Input metal-corrupted image [B, 1, H, W]
            XLI: Linear interpolation result [B, 1, H, W]
            mask: Metal artifact mask [B, 1, H, W]
            
        Returns:
            X0: Initial reconstruction
            ListX: List of reconstructed images at each stage
            ListA: List of estimated artifacts at each stage
            ListC: List of sparse coefficients at each stage
        """
        B, C, H, W = Y.shape
        
        # Feature extraction
        features = self.feature_extract(Y)  # [B, num_Q, H, W]
        
        # Initialize variables for ADMM
        X = torch.zeros_like(features)
        Z = torch.zeros_like(features)
        U = torch.zeros_like(features)
        
        # Lists to store intermediate results
        ListX = []
        ListA = []
        ListC = []
        
        # Unrolling optimization
        for i in range(self.S):
            # ADMM block
            X_new, Z_new, U_new, coeffs = self.admm_blocks[i](Y, XLI, mask, X, Z, U)
            
            # Update variables
            X, Z, U = X_new, Z_new, U_new
            
            # Store intermediate results
            ListC.append(coeffs)
            
            # Reconstruct image and artifact
            if i == 0:
                X0 = self.reconstruction(X)
            
            X_recon = self.reconstruction(X)
            A_est = self.artifact_estimation(X)
            
            ListX.append(X_recon)
            ListA.append(A_est)
        
        return X0, ListX, ListA, ListC