# training/ssl_augmentations.py
"""
Data augmentation strategies for SSL pre-training.
Provides dataset-specific augmentation pipelines for SimCLR.
"""

import torch
from torchvision import transforms
import numpy as np


class SimCLRTransform:
    """
    Transform that creates two augmented views of each image for SimCLR.
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        """Returns two augmented views of the input image."""
        return self.transform(x), self.transform(x)


def get_simclr_augmentation(dataset_name, image_size=32):
    """
    Get SimCLR augmentation pipeline for specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('CIFAR10', 'CIFAR100', 'SVHN', 'MNIST')
        image_size: Size of the images
        
    Returns:
        SimCLRTransform object that produces two views
    """
    
    if dataset_name in ['CIFAR10', 'CIFAR100']:
        # Full augmentation pipeline for CIFAR
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
    elif dataset_name == 'SVHN':
        # Reduced color augmentation for SVHN (numbers are color-sensitive)
        transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)  # Reduced intensity
            ], p=0.8),
            transforms.ToTensor(),
            transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
        ])
        
    elif dataset_name == 'MNIST':
        # Simple augmentations for MNIST
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ])
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return SimCLRTransform(transform)


class SSLDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for SSL that applies augmentations and returns two views.
    """
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index in the base dataset
        actual_idx = self.indices[idx]
        
        # Get image and label (label not used in SSL)
        image, _ = self.base_dataset[actual_idx]
        
        # Apply transform to get two views
        view1, view2 = self.transform(image)
        
        return view1, view2, idx  # Return idx for tracking
