"""
Federated training functions for self-supervised learning with autoencoders.
This module provides functionality to train a global autoencoder in a federated manner
after distributing data to clients for federated active learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T  # Added for contrastive augmentations
import numpy as np
from tqdm import tqdm
import os
import copy


def train_local_autoencoder(model, train_loader, device, epochs=1, lr=1e-3):
    """
    Train a local autoencoder model on a client's dataset
    
    Args:
        model: The autoencoder model to train
        train_loader: DataLoader for client's data
        device: The device to use for training
        epochs: Number of local epochs to train
        lr: Learning rate
        
    Returns:
        The trained autoencoder model and its average loss
    """
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train the model
    model.to(device)
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in train_loader:
            # Move data to device
            inputs = inputs.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructions, _ = model(inputs)
            
            # Compute loss
            loss = criterion(reconstructions, inputs)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update stats
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Local Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        total_loss += running_loss
    
    # Calculate average loss for this client
    avg_loss = total_loss / (total_samples * epochs)
    
    return model, avg_loss


# ADDED: New contrastive version of the local autoencoder training function
def train_local_autoencoder_contrastive(model, train_loader, device, global_model=None, epochs=1, lr=1e-3, temperature=0.5, mu=0.01):
    """
    Train a local autoencoder with simple contrastive loss and optional proximal regularization
    
    Args:
        model: The autoencoder model to train
        train_loader: DataLoader for client's data
        device: The device to use for training
        global_model: The global model for proximal term regularization (optional)
        epochs: Number of local epochs to train
        lr: Learning rate
        temperature: Temperature parameter for contrastive loss
        mu: Proximal term weight for regularization
        
    Returns:
        The trained autoencoder model and its average loss
    """
    # ADDED: Simple augmentation transforms for contrastive learning
    augment1 = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomHorizontalFlip()
    ])
    
    augment2 = T.Compose([
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip()
    ])
    
    # Set up optimizer and loss functions
    optimizer = optim.Adam(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()
    
    # Save global model parameters for proximal term if provided
    global_params = None
    if global_model is not None:
        global_params = [p.clone().detach() for p in global_model.parameters()]
    
    # Train the model
    model.to(device)
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, _ in train_loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            
            # ADDED: Create two augmented versions of the inputs for contrastive learning
            # We need to process each image individually to apply the transforms
            inputs_aug1 = []
            inputs_aug2 = []
            
            for i in range(batch_size):
                img = inputs[i].cpu()
                # Convert to PIL for transforms then back to tensor
                img_pil = T.ToPILImage()(img)
                inputs_aug1.append(T.ToTensor()(augment1(img_pil)))
                inputs_aug2.append(T.ToTensor()(augment2(img_pil)))
            
            # Stack back into batches and normalize
            inputs_aug1 = torch.stack(inputs_aug1).to(device)
            inputs_aug2 = torch.stack(inputs_aug2).to(device)
            
            # Normalize if transforms removed the normalization
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
            inputs_aug1 = (inputs_aug1 - mean) / std
            inputs_aug2 = (inputs_aug2 - mean) / std
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass for reconstruction (original images)
            reconstructions, z = model(inputs)
            recon_loss = recon_criterion(reconstructions, inputs)
            
            # ADDED: Forward pass for the augmented views to get latent representations
            _, z1 = model(inputs_aug1)
            _, z2 = model(inputs_aug2)
            
            # ADDED: Simple contrastive loss calculation (InfoNCE-style)
            # Normalize the embeddings
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            # Calculate similarity matrix
            similarity_matrix = torch.matmul(z1, z2.T) / temperature
            
            # Positive pairs are the diagonal elements (same image, different augmentations)
            # Labels for these positive pairs
            labels = torch.arange(batch_size).to(device)
            
            # Calculate loss in both directions
            loss_1 = F.cross_entropy(similarity_matrix, labels)
            loss_2 = F.cross_entropy(similarity_matrix.T, labels)
            contrastive_loss = (loss_1 + loss_2) / 2
            
            # ADDED: Calculate proximal term if global model is provided
            proximal_term = 0.0
            if global_params is not None:
                local_params = list(model.parameters())
                proximal_term = mu * sum([((local_p - global_p.to(local_p.device)) ** 2).sum()
                          for local_p, global_p in zip(local_params, global_params)])

            
            # Combined loss: reconstruction + contrastive + proximal
            loss = recon_loss + contrastive_loss + proximal_term
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update stats
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Local Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        total_loss += running_loss
    
    # Calculate average loss for this client
    avg_loss = total_loss / (total_samples * epochs)
    
    return model, avg_loss


def federated_train_autoencoder(
    init_model, 
    client_train_loaders, 
    device, 
    local_epochs=1, 
    rounds=3, 
    lr=1e-3,
    use_contrastive=True,  # ADDED: Flag to toggle contrastive learning
    temperature=0.5,       # ADDED: Temperature for contrastive loss
    mu=0.01,               # ADDED: Proximal term weight
    save_dir='models/ssl/global_encoder'
):
    """
    Train a global autoencoder model in a federated manner.
    Each client trains a local model, then models are aggregated on the server.
    
    Args:
        init_model: The initial autoencoder model
        client_train_loaders: List of DataLoaders for each client
        device: The device to use for training
        local_epochs: Number of local training epochs per round
        rounds: Number of federated rounds
        lr: Learning rate for local training
        use_contrastive: Whether to use contrastive learning
        temperature: Temperature parameter for contrastive loss
        mu: Proximal term weight for regularization
        save_dir: Directory to save the trained model
        
    Returns:
        The trained global autoencoder model
    """
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    num_clients = len(client_train_loaders)
    global_model = init_model
    
    best_loss = float('inf')
    
    # ADDED: Print whether using contrastive learning
    if use_contrastive:
        print(f"Starting federated autoencoder training with contrastive learning for {rounds} rounds...")
    else:
        print(f"Starting federated autoencoder training for {rounds} rounds...")
    
    for round_idx in range(rounds):
        print(f"\n--- Round {round_idx+1}/{rounds} ---")
        
        # Initialize collection of local models and their weights
        local_models = []
        sample_sizes = []
        round_loss = 0.0
        
        # Train local models on each client
        for client_idx, train_loader in enumerate(client_train_loaders):
            print(f"Training client {client_idx+1}/{num_clients}")
            
            # Initialize model with global parameters
            local_model = copy.deepcopy(global_model)
            
            # MODIFIED: Choose appropriate training function based on use_contrastive flag
            if use_contrastive:
                # Train with contrastive learning
                local_model, local_loss = train_local_autoencoder_contrastive(
                    model=local_model,
                    train_loader=train_loader,
                    device=device,
                    global_model=global_model,  # Pass global model for proximal term
                    epochs=local_epochs,
                    lr=lr,
                    temperature=temperature,
                    mu=mu
                )
            else:
                # Original training without contrastive loss
                local_model, local_loss = train_local_autoencoder(
                    model=local_model,
                    train_loader=train_loader,
                    device=device,
                    epochs=local_epochs,
                    lr=lr
                )
            
            # Store the trained model and its weight (proportional to dataset size)
            local_models.append(local_model.state_dict())
            sample_sizes.append(len(train_loader.dataset))
            round_loss += local_loss * len(train_loader.dataset)
            
            print(f"Client {client_idx+1} completed training with loss: {local_loss:.6f}")
        
        # Calculate weighted average loss for this round
        total_samples = sum(sample_sizes)
        avg_round_loss = round_loss / total_samples
        
        # Aggregate models (weighted average of parameters)
        global_model_dict = copy.deepcopy(global_model.state_dict())
        
        for key in global_model_dict.keys():
            # Reset parameters
            global_model_dict[key] = torch.zeros_like(global_model_dict[key])
            
            # Weighted sum of parameters across clients
            for client_idx, local_model_dict in enumerate(local_models):
                weight = sample_sizes[client_idx] / total_samples
                global_model_dict[key] += local_model_dict[key] * weight
        
        # Update global model
        global_model.load_state_dict(global_model_dict)
        
        print(f"Round {round_idx+1} completed with average loss: {avg_round_loss:.6f}")
        
        # Save the model if it's the best so far
        if avg_round_loss < best_loss:
            best_loss = avg_round_loss
            torch.save({
                'round': round_idx,
                'model_state_dict': global_model.state_dict(),
                'loss': best_loss,
            }, os.path.join(save_dir, 'best_fed_autoencoder.pt'))
            print(f"Saved model with loss: {best_loss:.6f}")
    
    # Save the final model
    torch.save({
        'round': rounds,
        'model_state_dict': global_model.state_dict(),
        'loss': avg_round_loss,
    }, os.path.join(save_dir, 'final_fed_autoencoder.pt'))
    
    print(f"Federated autoencoder training completed. Final loss: {avg_round_loss:.6f}")
    
    # Load the best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_fed_autoencoder.pt'))
    global_model.load_state_dict(checkpoint['model_state_dict'])
    
    return global_model
