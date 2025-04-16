"""
Simple autoencoder model for self-supervised learning in federated active learning.
This autoencoder is trained on all data before partitioning to clients.

[ADDED] This entire file is new - created to implement a global autoencoder that learns
representations from all data before federated active learning begins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    """
    [ADDED] Convolutional Autoencoder for CIFAR-10 images
    This model is trained on the entire dataset before partitioning to clients
    to learn useful representations that can be used in active learning
    """
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        # [ADDED] Encoder layers - convolutional architecture for image data
        self.encoder = nn.Sequential(
            # 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 128x4x4 -> 256x2x2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # [ADDED] FC layers for latent space - bottleneck representation
        self.fc_encoder = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, 256 * 2 * 2)
        
        # [ADDED] Decoder layers - mirror of encoder for reconstruction
        self.decoder = nn.Sequential(
            # 256x2x2 -> 128x4x4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 128x4x4 -> 64x8x8
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 64x8x8 -> 32x16x16
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32x16x16 -> 3x32x32
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output normalized between -1 and 1
        )
        
    def encode(self, x):
        """[ADDED] Encode input to latent representation"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        z = self.fc_encoder(x)
        return z
    
    def decode(self, z):
        """[ADDED] Decode latent representation to reconstruction"""
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 256, 2, 2)  # Reshape
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        """[ADDED] Forward pass through the autoencoder"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_features(self, x):
        """[ADDED] Extract features for downstream tasks"""
        with torch.no_grad():
            z = self.encode(x)
        return z


def create_autoencoder(latent_dim=128):
    """[ADDED] Factory function to create the autoencoder"""
    return ConvAutoencoder(latent_dim=latent_dim)
