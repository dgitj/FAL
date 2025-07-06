# models/ssl_models.py
"""
SSL model components for federated self-supervised pre-training.
Provides encoders and projection heads for SimCLR-style training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.preact_resnet import BasicBlock


class PreActResNetEncoder(nn.Module):
    """
    Encoder part of PreActResNet without the final classification layer.
    Extracts features that can be used for SSL pre-training.
    """
    def __init__(self, block, num_blocks, num_classes=10, dataset='cifar'):
        super(PreActResNetEncoder, self).__init__()
        self.in_planes = 64
        self.dataset = dataset
        
        if dataset == 'mnist':
            # MNIST is 1 channel
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.output_dim = 512 * block.expansion
        else:
            # CIFAR is 3 channels
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.output_dim = 512 * block.expansion
            
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def create_encoder_cifar():
    """Create encoder for CIFAR datasets."""
    return PreActResNetEncoder(BasicBlock, [1, 1, 1, 1], dataset='cifar')


def create_encoder_mnist():
    """Create encoder for MNIST dataset."""
    return PreActResNetEncoder(BasicBlock, [1, 1, 1, 1], dataset='mnist')


class ProjectionHead(nn.Module):
    """
    Projection head for SSL pre-training.
    Maps encoder features to a lower dimensional space for contrastive learning.
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class PretrainedPreActResNet(nn.Module):
    """
    Combines a pre-trained encoder with a new classification head.
    Used after SSL pre-training to create the final model for active learning.
    """
    def __init__(self, encoder, num_classes):
        super(PretrainedPreActResNet, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, num_classes)
        
    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features


def create_model_with_pretrained_encoder_cifar(encoder, num_classes):
    """
    Create a complete model using pre-trained encoder for CIFAR.
    
    Args:
        encoder: Pre-trained encoder from SSL
        num_classes: Number of output classes
        
    Returns:
        Complete model with encoder + classification head
    """
    return PretrainedPreActResNet(encoder, num_classes)


def create_model_with_pretrained_encoder_mnist(encoder, num_classes):
    """
    Create a complete model using pre-trained encoder for MNIST.
    
    Args:
        encoder: Pre-trained encoder from SSL
        num_classes: Number of output classes
        
    Returns:
        Complete model with encoder + classification head
    """
    return PretrainedPreActResNet(encoder, num_classes)


class SimCLRModel(nn.Module):
    """
    Complete SimCLR model combining encoder and projection head.
    Used during SSL pre-training only.
    """
    def __init__(self, encoder, projection_head):
        super(SimCLRModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)
