"""
Utility functions for global encoder training and integration.
"""

import os
import torch
import numpy as np
from sklearn.cluster import KMeans

from models.ssl.contrastive_model import SimpleContrastiveLearning


def create_encoder_distribution_checkpoint(encoder, dataloader, num_clusters=10, device="cuda"):
    """
    Calculate and save clustering distribution from the encoder.
    
    Args:
        encoder (SimpleContrastiveLearning): Trained encoder model
        dataloader (DataLoader): Dataloader for feature extraction
        num_clusters (int): Number of clusters for K-means (usually = num_classes)
        device (str): Device to use for computation
        
    Returns:
        tuple: (kmeans, distribution) - KMeans model and class distribution
    """
    encoder.eval()
    features_list = []
    
    # Extract features
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            inputs = inputs.to(device)
            batch_features = encoder.get_features(inputs)
            features_list.append(batch_features.cpu().numpy())
    
    # Combine all features
    features = np.vstack(features_list)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_assignments = kmeans.fit_predict(features)
    
    # Calculate distribution
    counts = np.bincount(cluster_assignments, minlength=num_clusters)
    distribution = counts / counts.sum()
    
    # Create checkpoint directory
    distribution_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
        'distribution'
    )
    os.makedirs(distribution_dir, exist_ok=True)
    
    # Save model, kmeans and distribution
    checkpoint = {
        'model_state_dict': encoder.state_dict(),
        'kmeans_model': kmeans,
        'distribution': distribution,
    }
    
    checkpoint_path = os.path.join(distribution_dir, 'round_99.pt')
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Encoder, KMeans, and distribution saved to {checkpoint_path}")
    print("Distribution:")
    for i in range(num_clusters):
        print(f"  Cluster {i}: {distribution[i]:.4f}")
    
    return kmeans, distribution


def load_global_encoder(checkpoint_path=None, feature_dim=128, device="cuda"):
    """
    Load a trained global encoder from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the encoder checkpoint
        feature_dim (int): Feature dimension
        device (str): Device to load the model on
        
    Returns:
        SimpleContrastiveLearning: Loaded encoder model
    """
    if checkpoint_path is None:
        # Use default path
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
            'models', 'ssl', 'global_encoder', 'checkpoints', 'best_encoder.pt'
        )
    
    # Create model instance
    encoder = SimpleContrastiveLearning(feature_dim=feature_dim).to(device)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Global encoder loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    return encoder
