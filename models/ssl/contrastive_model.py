"""
Simple implementation of the contrastive learning model from ssl_fl.
This is a standalone implementation for FAL that doesn't require the ssl_fl project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class EncoderNetwork(nn.Module):
    """
    Simple encoder network for contrastive learning
    """
    def __init__(self, feature_dim=128):
        super(EncoderNetwork, self).__init__()
        
        # Use ResNet18 as the encoder backbone
        resnet = resnet18(pretrained=False)
        # Modify for CIFAR10 - change first conv layer
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()  # Remove maxpool as CIFAR10 is small
        
        # Remove the final FC layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return F.normalize(z, dim=1)  # L2 normalization
    
    def get_features(self, x):
        h = self.encoder(x)
        return torch.flatten(h, 1)  # Return features before projection


class SimpleContrastiveLearning(nn.Module):
    """
    A simple contrastive learning framework
    """
    def __init__(self, feature_dim=128, temperature=0.5):
        super(SimpleContrastiveLearning, self).__init__()
        self.encoder = EncoderNetwork(feature_dim)
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
        return self.encoder.get_features(x)
