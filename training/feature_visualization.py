"""
Feature Space Visualization Module for FAL

This module provides functions to visualize the feature space using t-SNE
to evaluate the effectiveness of contrastive learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def visualize_feature_space(model, dataloader, device, title="Feature Space Visualization", 
                           output_path='feature_viz.png', max_samples=1000):
    """
    Visualize the feature space using t-SNE to evaluate feature separation.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader with test data
        device (torch.device): Device to run evaluation on
        title (str): Title for the visualization
        output_path (str): Path to save the visualization
        max_samples (int): Maximum number of samples to use for visualization
    
    Returns:
        dict: Metrics related to feature separation
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Extract features and labels
    features_list = []
    labels_list = []
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if sample_count >= max_samples:
                break
                
            inputs = inputs.to(device)
            # Get feature representations from model
            _, features = model(inputs)
            
            # Handle case where features is a list
            if isinstance(features, list):
                features = features[-1]  # Use the last layer features
            
            # Normalize features
            features = F.normalize(features, p=2, dim=1)
            
            # Add to lists
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
            
            sample_count += inputs.size(0)
    
    # Concatenate all features and labels
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    
    # Limit samples if needed
    if features.shape[0] > max_samples:
        indices = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Apply t-SNE for dimensionality reduction
    print(f"Running t-SNE on {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
    features_2d = tsne.fit_transform(features)
    
    # Compute metrics (optional)
    metrics = {}
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Get unique classes
    unique_classes = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_classes)))
    
    # Plot each class with a different color
    for i, class_idx in enumerate(unique_classes):
        idx = labels == class_idx
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], 
                   color=colors[i], label=f'Class {class_idx}',
                   alpha=0.7, s=50)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    
    return metrics

def compare_feature_spaces(models_dict, dataloader, device, output_dir='visualizations'):
    """
    Compare feature spaces of multiple models side by side.
    
    Args:
        models_dict (dict): Dictionary of models to compare (name -> model)
        dataloader (DataLoader): DataLoader with test data
        device (torch.device): Device to run evaluation on
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for name, model in models_dict.items():
        output_path = os.path.join(output_dir, f"features_{name}.png")
        visualize_feature_space(
            model, 
            dataloader, 
            device,
            title=f"Feature Space - {name}",
            output_path=output_path
        )
    
    print(f"All visualizations saved to {output_dir}")
