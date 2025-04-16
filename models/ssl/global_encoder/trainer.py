"""
Global Encoder Trainer for Self-Supervised Learning

This module trains a global encoder using contrastive learning techniques on client data
before active learning cycles begin. The global encoder can be used to extract features
that help with more informed sample selection during active learning.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from models.ssl.contrastive_model import SimpleContrastiveLearning


class ContrastiveTransform:
    """
    Provides two random transformations of the same image for contrastive learning.
    """
    def __init__(self, base_transform, size=32):
        # Base transform that's common to all transformations
        self.base_transform = base_transform
        
        # Additional stochastic augmentations for contrastive views
        self.aug_transform = T.Compose([
            T.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
        ])
        
    def __call__(self, x):
        # Apply augmentations to create two views
        view1 = self.aug_transform(x)
        view2 = self.aug_transform(x)
        
        # Apply base transform to both views
        return self.base_transform(view1), self.base_transform(view2)


class GlobalEncoderTrainer:
    """
    Trains a global encoder using contrastive learning on federated client data.
    """
    def __init__(self, device, config, client_data_indices, feature_dim=128, temperature=0.5):
        """
        Initialize the global encoder trainer.
        
        Args:
            device (torch.device): Device to run training on (cuda/cpu)
            config (module): Configuration module with parameters
            client_data_indices (list): List of client data indices after partitioning
            feature_dim (int): Dimension of feature vectors
            temperature (float): Temperature parameter for contrastive loss
        """
        self.device = device
        self.config = config
        self.client_data_indices = client_data_indices
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Create model
        self.model = SimpleContrastiveLearning(feature_dim=feature_dim, temperature=temperature).to(device)
        
        # Create directory to save encoder
        self.save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                                    'models', 'ssl', 'global_encoder', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def prepare_contrastive_dataset(self):
        """
        Prepare dataset with contrastive transforms for SSL training.
        
        Returns:
            DataLoader: DataLoader with contrastive pairs
        """
        # Base CIFAR10 transformations (normalization)
        base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
        # Contrastive transforms that generate two views of each image
        contrastive_transform = ContrastiveTransform(base_transform)
        
        # Dataset directory
        cifar10_dataset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
            self.config.DATA_ROOT
        )
        
        # Create a dataset with PILImage output (not tensor yet)
        ssl_dataset = CIFAR10(
            cifar10_dataset_dir, 
            train=True, 
            download=True, 
            transform=contrastive_transform
        )
        
        # Combine all client data indices for global training
        all_indices = []
        for client_indices in self.client_data_indices:
            all_indices.extend(client_indices)
        
        # Create a contrastive dataloader with all client data
        ssl_loader = DataLoader(
            ssl_dataset,
            batch_size=self.config.BATCH,
            sampler=torch.utils.data.SubsetRandomSampler(all_indices),
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        return ssl_loader
    
    def contrastive_loss(self, features_1, features_2):
        """
        Compute the contrastive loss (InfoNCE/NT-Xent) for self-supervised learning.
        
        Args:
            features_1 (torch.Tensor): Normalized feature vectors from first augmented view
            features_2 (torch.Tensor): Normalized feature vectors from second augmented view
            
        Returns:
            torch.Tensor: Contrastive loss value
        """
        batch_size = features_1.shape[0]
        
        # Concatenate features from both views
        features = torch.cat([features_1, features_2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        # Mask out self-similarity
        sim_matrix.fill_diagonal_(-float('inf'))
        
        # Create positive pair indices
        pos_idx = torch.arange(batch_size, device=self.device)
        pos_idx_1 = pos_idx
        pos_idx_2 = pos_idx + batch_size
        
        # Get positive similarities
        pos_sim_1 = sim_matrix[pos_idx_1, pos_idx_2]
        pos_sim_2 = sim_matrix[pos_idx_2, pos_idx_1]
        
        # InfoNCE loss
        loss_1 = -pos_sim_1 + torch.logsumexp(sim_matrix[pos_idx_1], dim=1)
        loss_2 = -pos_sim_2 + torch.logsumexp(sim_matrix[pos_idx_2], dim=1)
        
        # Average over batch
        loss = (loss_1.mean() + loss_2.mean()) / 2
        
        return loss
    
    def train(self, epochs=100):
        """
        Train the global encoder using contrastive learning.
        
        Args:
            epochs (int): Number of training epochs
            
        Returns:
            SimpleContrastiveLearning: Trained model
        """
        print("\n===== Training Global Encoder with Contrastive Learning =====")
        print(f"Using {len(self.client_data_indices)} clients' data")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Training for {epochs} epochs\n")
        
        # Prepare contrastive dataset
        ssl_loader = self.prepare_contrastive_dataset()
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-6)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            running_loss = 0.0
            
            for batch_idx, ((img1, img2), _) in enumerate(ssl_loader):
                img1, img2 = img1.to(self.device), img2.to(self.device)
                
                # Forward pass to get embeddings
                z1 = self.model(img1)
                z2 = self.model(img2)
                
                # Compute contrastive loss
                loss = self.contrastive_loss(z1, z2)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Print progress
                if (batch_idx + 1) % 20 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(ssl_loader)}], "
                          f"Loss: {loss.item():.4f}")
            
            # Calculate average loss for epoch
            avg_loss = running_loss / len(ssl_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint if loss improved
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(self.save_dir, 'best_encoder.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
            
            # Save model regularly
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'encoder_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
        
        # Load best model
        best_checkpoint_path = os.path.join(self.save_dir, 'best_encoder.pt')
        checkpoint = torch.load(best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"\nGlobal encoder training completed. Best loss: {best_loss:.4f}")
        print(f"Best model saved at {best_checkpoint_path}")
        print("==========================================================\n")
        
        return self.model
