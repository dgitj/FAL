"""
Training functions for self-supervised learning with autoencoders.

[ADDED] This entire file is new - created to implement training functionality
for the global autoencoder before distributing data to clients for
federated active learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os


def train_global_autoencoder(model, dataset, device, 
                             batch_size=128, epochs=10, lr=1e-3, 
                             save_dir='models/ssl/global_encoder'):
    """
    [ADDED] Train a global autoencoder model on the entire dataset
    
    This function is called before data distribution to clients,
    implementing a simple SSL step that learns representations
    from all data that might be useful for subsequent active learning.
    
    Args:
        model: The autoencoder model to train
        dataset: The dataset to train on
        device: The device to use for training
        batch_size: Batch size for training
        epochs: Number of epochs to train
        lr: Learning rate
        save_dir: Directory to save the trained model
        
    Returns:
        The trained autoencoder model
    """
    # [ADDED] Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # [ADDED] Create data loader for all data
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # [ADDED] Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # [ADDED] Train the model
    model.to(device)
    model.train()
    
    best_loss = float('inf')
    
    print(f"Starting global autoencoder training for {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, _ in pbar:
            # [ADDED] Move data to device
            inputs = inputs.to(device)
            
            # [ADDED] Zero the gradients
            optimizer.zero_grad()
            
            # [ADDED] Forward pass - autoencoder reconstructs the input
            reconstructions, _ = model(inputs)
            
            # [ADDED] Compute loss - how well the autoencoder can reconstruct
            loss = criterion(reconstructions, inputs)
            
            # [ADDED] Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # [ADDED] Update stats
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        # [ADDED] Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        # [ADDED] Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(save_dir, 'best_autoencoder.pt'))
            print(f"Saved model with loss: {best_loss:.6f}")
    
    # [ADDED] Save the final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, os.path.join(save_dir, 'final_autoencoder.pt'))
    
    print(f"Global autoencoder training completed. Final loss: {epoch_loss:.6f}")
    
    # [ADDED] Load the best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_autoencoder.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
