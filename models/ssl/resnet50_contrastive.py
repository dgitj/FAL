"""
Implementation of a contrastive learning model using ResNet-18 backbone
designed to work directly with pre-trained ResNet-18 checkpoints.
Note: This is not a ResNet-50 as originally thought, but a ResNet-18 based on the key structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    """
    ResNet encoder network that matches the checkpoint structure exactly
    """
    def __init__(self, feature_dim=128):
        super(ResNetEncoder, self).__init__()
        
        # Initialize the encoder with ResNet18 (based on the checkpoint structure)
        self.encoder = resnet18(pretrained=False)
        
        # Modify for CIFAR10 - change first conv layer to handle smaller images
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()  # Remove maxpool as CIFAR10 is small
        
        # Remove the final FC layer (classification head)
        self.encoder.fc = nn.Identity()
        
        # Separate projector for contrastive learning that matches the checkpoint structure
        self.projector = nn.Sequential(
            nn.Linear(512, 512),  # ResNet18's output is 512-dimensional
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)  # L2 normalization
    
    def get_features(self, x):
        return self.encoder(x)


class ContrastiveModel(nn.Module):
    """
    Contrastive learning model that exactly matches the checkpoint structure
    """
    def __init__(self, feature_dim=128, temperature=0.5):
        super(ContrastiveModel, self).__init__()
        self.encoder = ResNetEncoder(feature_dim)
        self.temperature = temperature
        
    def forward(self, x1, x2=None):
        # Handle both single input and dual input cases
        if x2 is None:
            # If only one input is provided, just get its representation
            z1 = self.encoder(x1)
            return z1
        else:
            # If two inputs are provided, get representations for both
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            return z1, z2
    
    def get_features(self, x):
        """Get feature representation before the projection head"""
        return self.encoder.get_features(x)
