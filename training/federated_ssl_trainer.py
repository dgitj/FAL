# training/federated_ssl_trainer.py
"""
Federated SSL trainer implementing vanilla SimCLR for federated learning.
Performs local contrastive learning on each client and aggregates encoder weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from torch.utils.data import DataLoader

from models.ssl_models import create_encoder_cifar, create_encoder_mnist, ProjectionHead, SimCLRModel
from training.ssl_augmentations import get_simclr_augmentation, SSLDataset


class FederatedSSLTrainer:
    """
    Implements vanilla federated SimCLR pre-training.
    Each client trains locally with SimCLR loss, then encoders are aggregated.
    """
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.temperature = config.SSL_TEMPERATURE
        
    def federated_ssl_pretrain(self, data_splits, base_dataset, trial_seed):
        """
        Main federated SSL pre-training loop.
        
        Args:
            data_splits: List of data indices for each client
            base_dataset: The base dataset (e.g., CIFAR10)
            trial_seed: Random seed for reproducibility
            
        Returns:
            Pre-trained encoder model
        """
        print("\n" + "="*50)
        print("Starting Federated SimCLR Pre-training")
        print(f"SSL Rounds: {self.config.SSL_ROUNDS}")
        print(f"Local Epochs per Round: {self.config.SSL_LOCAL_EPOCHS}")
        print(f"Batch Size: {self.config.SSL_BATCH_SIZE}")
        print("="*50 + "\n")
        
        # Set random seed
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)
        
        # Create encoder based on dataset
        if self.config.DATASET == "MNIST":
            encoder = create_encoder_mnist()
        else:
            encoder = create_encoder_cifar()
        encoder = encoder.to(self.device)
        
        # Initialize client models
        client_encoders = []
        client_projection_heads = []
        
        for c in range(self.config.CLIENTS):
            # Each client gets a copy of the encoder
            client_encoder = copy.deepcopy(encoder).to(self.device)
            client_encoders.append(client_encoder)
            
            # Each client gets its own projection head
            projection_head = ProjectionHead(
                input_dim=encoder.output_dim,
                hidden_dim=encoder.output_dim,
                output_dim=self.config.SSL_PROJECTION_DIM
            ).to(self.device)
            client_projection_heads.append(projection_head)
        
        # Get augmentation pipeline
        ssl_transform = get_simclr_augmentation(self.config.DATASET)
        
        # Create data loaders for each client
        client_loaders = []
        for c in range(self.config.CLIENTS):
            ssl_dataset = SSLDataset(
                base_dataset=base_dataset,
                indices=data_splits[c],
                transform=ssl_transform
            )
            
            # Ensure batch size is not larger than dataset
            effective_batch_size = min(self.config.SSL_BATCH_SIZE, len(data_splits[c]))
            if effective_batch_size < self.config.SSL_BATCH_SIZE:
                print(f"Warning: Client {c} has only {len(data_splits[c])} samples, "
                      f"reducing batch size from {self.config.SSL_BATCH_SIZE} to {effective_batch_size}")
            
            client_loader = DataLoader(
                ssl_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=0,  # Set to 0 for Windows compatibility
                drop_last=True if effective_batch_size > 1 else False,  # Only drop last if we have more than 1 batch
                pin_memory=True
            )
            client_loaders.append(client_loader)
        
        # Federated SSL training rounds
        for round_idx in range(self.config.SSL_ROUNDS):
            print(f"\n--- SSL Round {round_idx + 1}/{self.config.SSL_ROUNDS} ---")
            
            # Client selection (use all clients for SSL)
            selected_clients = list(range(self.config.CLIENTS))
            
            # Local training on each client
            client_losses = []
            for c in selected_clients:
                avg_loss = self._local_ssl_train(
                    client_encoders[c],
                    client_projection_heads[c],
                    client_loaders[c],
                    round_idx
                )
                client_losses.append(avg_loss)
                print(f"Client {c}: Avg Loss = {avg_loss:.4f}")
            
            # Aggregate encoder weights (simple averaging)
            self._aggregate_encoders(encoder, client_encoders, selected_clients, data_splits)
            
            # Broadcast aggregated encoder back to clients
            for c in selected_clients:
                client_encoders[c].load_state_dict(encoder.state_dict())
            
            print(f"Round {round_idx + 1} - Average Loss: {np.mean(client_losses):.4f}")
        
        print("\n" + "="*50)
        print("Federated SSL Pre-training Completed!")
        print("="*50 + "\n")
        
        return encoder
    
    def _local_ssl_train(self, encoder, projection_head, data_loader, round_idx):
        """
        Local SimCLR training on one client.
        
        Args:
            encoder: Client's encoder model
            projection_head: Client's projection head
            data_loader: Client's data loader
            round_idx: Current round index
            
        Returns:
            Average loss for this client
        """
        # Create combined model
        model = SimCLRModel(encoder, projection_head).to(self.device)
        model.train()
        
        # Optimizer (using higher learning rate for SSL)
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.SSL_LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WDECAY
        )
        
        # Learning rate schedule (cosine annealing within each round)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.SSL_LOCAL_EPOCHS * len(data_loader)
        )
        
        total_loss = 0
        num_batches = 0
        
        # Local training epochs
        for epoch in range(self.config.SSL_LOCAL_EPOCHS):
            for batch_idx, (view1, view2, _) in enumerate(data_loader):
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                # Get projections for both views
                z1 = model(view1)
                z2 = model(view2)
                
                # Compute SimCLR loss
                loss = self._simclr_loss(z1, z2)
                
                # Debug: Check if loss is reasonable
                if batch_idx == 0 and epoch == 0:
                    print(f"  Initial loss: {loss.item():.4f}")
                    print(f"  z1 mean: {z1.mean().item():.4f}, std: {z1.std().item():.4f}")
                    print(f"  z2 mean: {z2.mean().item():.4f}, std: {z2.std().item():.4f}")
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _simclr_loss(self, z1, z2):
        """
        Compute NT-Xent loss for SimCLR.
        
        Args:
            z1: Projections from view 1 [batch_size, projection_dim]
            z2: Projections from view 2 [batch_size, projection_dim]
            
        Returns:
            Contrastive loss
        """
        batch_size = z1.shape[0]
        
        # Check for feature collapse
        if torch.isnan(z1).any() or torch.isnan(z2).any():
            print("WARNING: NaN detected in projections!")
            
        # Concatenate projections
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, projection_dim]
        
        # Compute similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim = sim / self.temperature
        
        # Debug: Check similarity matrix
        if torch.isnan(sim).any():
            print("WARNING: NaN in similarity matrix!")
            print(f"z norms: min={z.norm(dim=1).min():.4f}, max={z.norm(dim=1).max():.4f}")
        
        # Create labels - positive pairs are (i, i+batch_size)
        labels = torch.cat([torch.arange(batch_size) + batch_size,
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(self.device)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        sim.masked_fill_(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim, labels)
        
        # Debug: Check if loss is reasonable
        if loss.item() > 10 or loss.item() < 0.1:
            print(f"WARNING: Unusual loss value: {loss.item():.4f}")
            print(f"Similarity matrix stats: min={sim.min():.4f}, max={sim.max():.4f}, mean={sim.mean():.4f}")
        
        return loss
    
    def _aggregate_encoders(self, global_encoder, client_encoders, selected_clients, data_splits):
        """
        Aggregate encoder weights from selected clients.
        Simple weighted averaging based on data size.
        
        Args:
            global_encoder: Global encoder to update
            client_encoders: List of client encoders
            selected_clients: Indices of participating clients
            data_splits: Data splits to get client data sizes
        """
        # Get data sizes for weighting
        total_data = sum(len(data_splits[c]) for c in selected_clients)
        weights = [len(data_splits[c]) / total_data for c in selected_clients]
        
        # Initialize global state dict
        global_state = global_encoder.state_dict()
        
        # Aggregate each parameter
        for key in global_state.keys():
            # Handle different parameter types
            if 'num_batches_tracked' in key:
                # For batch norm tracking stats, just use the first client's value
                # (these are not trainable parameters)
                global_state[key] = client_encoders[selected_clients[0]].state_dict()[key].clone()
            else:
                # For regular parameters, do weighted averaging
                # Initialize with zeros of the same type
                global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
                
                # Weighted sum of client parameters
                for idx, c in enumerate(selected_clients):
                    client_state = client_encoders[c].state_dict()
                    global_state[key] += weights[idx] * client_state[key].float()
                
                # Convert back to original dtype if needed
                if global_encoder.state_dict()[key].dtype != torch.float32:
                    global_state[key] = global_state[key].to(global_encoder.state_dict()[key].dtype)
        
        # Update global encoder
        global_encoder.load_state_dict(global_state)


def perform_federated_ssl_pretraining(data_splits, config, device, trial_seed, base_dataset):
    """
    Helper function to perform federated SSL pre-training.
    
    Args:
        data_splits: List of data indices for each client
        config: Configuration object
        device: Torch device
        trial_seed: Random seed
        base_dataset: Base dataset for SSL training
        
    Returns:
        Pre-trained encoder
    """
    trainer = FederatedSSLTrainer(config, device)
    pretrained_encoder = trainer.federated_ssl_pretrain(
        data_splits=data_splits,
        base_dataset=base_dataset,
        trial_seed=trial_seed
    )
    
    return pretrained_encoder
