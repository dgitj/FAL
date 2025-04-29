"""
Federated Learning Trainer Module
Manages the training process for federated active learning.
"""

import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Beta


def tuned_contrastive_loss(features, labels, temperature=0.5, lambda_weight=1.0, 
                          hard_mining_ratio=0.5, adaptive_temp=True):
    """
    Computes the Tuned Contrastive Loss (TCL) with adaptive temperature and hard negative mining.
    
    Args:
        features (torch.Tensor): Feature vectors from the model [batch_size, feature_dim]
        labels (torch.Tensor): Ground truth labels [batch_size]
        temperature (float): Base temperature parameter for scaling similarities
        lambda_weight (float): Weight for TCL when combined with cross-entropy
        hard_mining_ratio (float): Ratio of hard negatives to prioritize
        adaptive_temp (bool): Whether to use adaptive temperature
        
    Returns:
        torch.Tensor: Computed TCL loss
    """
    # Handle case where features is a list (from some model architectures)
    if isinstance(features, list):
        features = features[-1]  # Use the last layer features
        
    batch_size = features.size(0)
    device = features.device
    
    # Normalize features (use L2 normalization)
    normalized_features = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
    
    # Create masks for positive and negative pairs
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    positive_mask = labels_matrix.float() - torch.eye(batch_size, device=device)
    negative_mask = (~labels_matrix).float()
    
    # Adaptive temperature based on feature similarity (optional)
    temp = temperature
    if adaptive_temp:
        # Compute average similarity of positive pairs for each anchor
        pos_sim = (similarity_matrix * positive_mask).sum(1) / (positive_mask.sum(1) + 1e-8)
        # Adjust temperature: harder positives (lower similarity) get lower temperature
        temp = temperature * (1.0 - 0.5 * (1.0 - pos_sim))
        temp = temp.unsqueeze(1)  # Shape for broadcasting
    
    # Scale similarity scores
    exp_sim = torch.exp(similarity_matrix / temp)
    
    # Hard negative mining
    if hard_mining_ratio < 1.0:
        # For each anchor, find the hardest negatives (highest similarity)
        neg_sim = similarity_matrix * negative_mask
        neg_sim[negative_mask == 0] = float('-inf')  # Mask out positives
        
        # Get indices of top k% hardest negatives
        num_hard_negatives = max(1, int(negative_mask.sum(1).min().item() * hard_mining_ratio))
        _, hard_indices = torch.topk(neg_sim, k=num_hard_negatives, dim=1)
        
        # Create mask for hard negatives
        hard_negative_mask = torch.zeros_like(negative_mask)
        for i in range(batch_size):
            hard_negative_mask[i, hard_indices[i]] = 1.0
        
        # Apply hard negative mask
        negative_mask = hard_negative_mask
    
    # Compute contrastive loss
    pos_exp_sum = (exp_sim * positive_mask).sum(1, keepdim=True)
    neg_exp_sum = (exp_sim * negative_mask).sum(1, keepdim=True)
    
    # Handle cases with no positives
    denominator = pos_exp_sum + neg_exp_sum
    
    # Add epsilon for numerical stability
    epsilon = 1e-12
    loss = -torch.log((pos_exp_sum + epsilon) / (denominator + epsilon))
    
    # Check for any numerical issues
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("[WARNING] NaN or Inf detected in contrastive loss")
        # Return a zero loss if there are numerical issues to avoid breaking training
        return torch.tensor(0.0, device=device)
    
    # Average loss over all samples with positives
    valid_samples = (positive_mask.sum(1) > 0).float()
    if valid_samples.sum() > 0:
        loss = (loss * valid_samples).sum() / (valid_samples.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=device)
    
    return loss * lambda_weight


def simple_contrastive_loss(features, labels, temperature=0.5):
    """
    Computes a simple contrastive loss that pulls together features from the same class
    and pushes apart features from different classes.
    
    Args:
        features (torch.Tensor): Feature vectors from the model [batch_size, feature_dim]
        labels (torch.Tensor): Ground truth labels [batch_size]
        temperature (float): Temperature parameter for scaling similarities
        
    Returns:
        torch.Tensor: Computed simple contrastive loss
    """
    # Handle case where features is a list (from some model architectures)
    if isinstance(features, list):
        features = features[-1]  # Use the last layer features
        
    batch_size = features.size(0)
    device = features.device
    
    # Normalize features (use L2 normalization)
    normalized_features = F.normalize(features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
    
    # Create masks for positive and negative pairs
    labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    positive_mask = labels_matrix.float() - torch.eye(batch_size, device=device)
    negative_mask = (~labels_matrix).float()
    
    # Scale similarity scores
    sim_matrix = similarity_matrix / temperature
    
    # Compute contrastive loss
    # For each anchor, get the average similarity with positives
    pos_pairs = sim_matrix * positive_mask
    pos_pairs_sum = pos_pairs.sum(1)
    num_pos_pairs = positive_mask.sum(1)
    
    # Compute mean similarity for positive pairs (with protection against no positives)
    valid_samples = (num_pos_pairs > 0).float()
    mean_pos_sim = torch.zeros_like(pos_pairs_sum)
    mean_pos_sim[valid_samples > 0] = pos_pairs_sum[valid_samples > 0] / num_pos_pairs[valid_samples > 0]
    
    # For each anchor, compute average similarity with negatives
    neg_pairs = sim_matrix * negative_mask
    neg_pairs_sum = neg_pairs.sum(1)
    num_neg_pairs = negative_mask.sum(1)
    mean_neg_sim = neg_pairs_sum / (num_neg_pairs + 1e-8)
    
    # The loss aims to make positive similarities larger than negative similarities
    # with a margin of 1.0
    pair_loss = torch.clamp(1.0 - mean_pos_sim + mean_neg_sim, min=0.0)
    
    # Average loss over all samples with positives
    if valid_samples.sum() > 0:
        loss = (pair_loss * valid_samples).sum() / (valid_samples.sum() + 1e-8)
    else:
        loss = torch.tensor(0.0, device=device)
    
    return loss


class FederatedTrainer:
    """
    Manages the federated learning training process with various
    client update strategies.
    """
    
    def __init__(self, device, config, logger=None):
        """
        Initialize the federated trainer.
        
        Args:
            device (torch.device): Device to run training on (cuda/cpu)
            config (module): Configuration module with training parameters
            logger (FederatedALLogger, optional): Logger for metrics
        """
        self.device = device
        self.config = config
        self.logger = logger
        self.iters = 0  # Training iteration counter
    
    def train(self, models, criterion, optimizers, schedulers, dataloaders, num_epochs, trial_seed, val_loader=None, max_rounds=None):
        """
        Main training loop for federated learning.
        
        Args:
            models (dict): Dictionary containing 'server' and 'clients' models
            criterion (nn.Module): Loss function
            optimizers (dict): Dictionary of optimizers for 'server' and 'clients'
            schedulers (dict): Dictionary of learning rate schedulers
            dataloaders (dict): Dictionary of dataloaders
            num_epochs (int): Number of local epochs per communication round
            trial_seed (int): Random seed for this trial
            val_loader (DataLoader, optional): Validation dataloader for convergence monitoring
            max_rounds (int, optional): Maximum communication rounds (overrides config.COMMUNICATION)
            
        Returns:
            dict: Training statistics including convergence information
        """
        print('>> Train a Model.')
        
        # Initialize training statistics
        stats = {
            'train_losses': [],
            'val_accuracies': [],
            'rounds_completed': 0,
            'best_val_accuracy': 0.0,
            'total_losses_per_round': []
        }
        
        # Track best validation accuracy
        best_val_acc = 0.0

        # Determine number of communication rounds
        num_rounds = max_rounds if max_rounds is not None else self.config.COMMUNICATION
        
        for comm in range(num_rounds):
            # Deterministic client selection
            rng = np.random.RandomState(trial_seed * 100 + comm * 10)

            if comm < self.config.COMMUNICATION - 1:
                selected_clients_id = rng.choice(self.config.CLIENTS, 
                                              int(self.config.CLIENTS * self.config.RATIO), 
                                              replace=False)
            else:
                selected_clients_id = range(self.config.CLIENTS)

            # Broadcast server model to selected clients
            server_state_dict = models['server'].state_dict()
            for c in selected_clients_id:
                models['clients'][c].load_state_dict(server_state_dict, strict=False)

            # Local updates
            start_time = time.time()
            epoch_losses = []
            
            for epoch in range(num_epochs):
                # Choose appropriate training method based on config
                if self.config.LOCAL_MODEL_UPDATE == "Vanilla":
                    epoch_loss = self._train_epoch_client_vanilla(
                        selected_clients_id, models, criterion, 
                        optimizers, dataloaders, trial_seed + epoch
                    )
                elif self.config.LOCAL_MODEL_UPDATE == "ContrastiveEntropy":
                    epoch_loss = self._train_epoch_client_contrastive_entropy(
                        selected_clients_id, models, criterion, 
                        optimizers, dataloaders, trial_seed + epoch,
                        tcl_temp=self.config.TCL_TEMPERATURE, 
                        tcl_lambda=self.config.TCL_LAMBDA,
                        tcl_hard_ratio=self.config.TCL_HARD_MINING_RATIO, 
                        tcl_adaptive_temp=self.config.TCL_ADAPTIVE_TEMP
                    )
                elif self.config.LOCAL_MODEL_UPDATE == "SimpleContrastive":
                    epoch_loss = self._train_epoch_client_simple_contrastive(
                        selected_clients_id, models, criterion, 
                        optimizers, dataloaders, trial_seed + epoch,
                        contrastive_temp=self.config.CONTRASTIVE_TEMPERATURE, 
                        contrastive_weight=self.config.CONTRASTIVE_WEIGHT
                    )
                else:  # KCFU
                    # KCFU method doesn't currently return loss, so we use a dummy value
                    epoch_loss = self._train_epoch_client_distil(
                        selected_clients_id, models, criterion, 
                        optimizers, dataloaders, comm, trial_seed + epoch
                    )
                
                epoch_losses.append(epoch_loss)
                
                # Update learning rate schedulers
                for c in selected_clients_id:
                    schedulers['clients'][c].step()
                
                print(f'Epoch: {epoch + 1}/{num_epochs} | '
                      f'Communication round: {comm + 1}/{num_rounds} | '
                      f'Avg Loss: {epoch_loss:.4f}')    
                      
            end_time = time.time()
            print(f'Average time per epoch: {(end_time-start_time)/num_epochs:.2f} seconds')

            # Aggregate client models to update the server model
            self._aggregate_models(models, selected_clients_id, self.data_num)
            
            # Calculate average training loss for this round
            avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            stats['train_losses'].append(avg_train_loss)
            stats['total_losses_per_round'].append(epoch_losses)
            
            # Evaluate on validation set if provided for convergence monitoring
            if val_loader is not None:
                val_acc = self.evaluate(models['server'], val_loader, mode='val')
                stats['val_accuracies'].append(val_acc)
                stats['rounds_completed'] = comm + 1
                
                print(f'Validation accuracy after round {comm+1}: {val_acc:.2f}%')
                
                # Track best validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    stats['best_val_accuracy'] = best_val_acc

        print('>> Training completed.')
        return stats
    
    def _train_epoch_client_vanilla(self, selected_clients_id, models, criterion, 
                                   optimizers, dataloaders, trial_seed):
        # Track total loss for convergence monitoring
        total_loss = 0.0
        total_batches = 0
        
        for c in selected_clients_id:
            # Set deterministic behavior for this client
            client_seed = (trial_seed * 100 + c * 10) % (2**31 - 1)
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(client_seed)

            model = models['clients'][c]
            model.train()
            
            for batch_idx, data in enumerate(dataloaders['train-private'][c]):
                # Use a batch-specific seed for reproducibility
                batch_seed = client_seed + batch_idx
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(batch_seed)
                
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                # Forward pass
                scores, _ = model(inputs)
                
                # Standard cross-entropy loss
                loss = torch.sum(criterion(scores, labels)) / labels.size(0)
                
                # Track loss for convergence monitoring
                total_loss += loss.item()
                total_batches += 1
                
                # Backward and optimize
                self.iters += 1
                optimizers['clients'][c].zero_grad()
                loss.backward()
                optimizers['clients'][c].step()
                
                # Log occasionally
                if (self.iters % 1000 == 0):
                    print(f'Client {c} | Batch {batch_idx} | Loss: {loss.item():.4f}')
        
        # Return average loss for this epoch
        avg_loss = total_loss / max(1, total_batches)
        return avg_loss
                    
    def _train_epoch_client_contrastive_entropy(self, selected_clients_id, models, criterion, 
                                               optimizers, dataloaders, trial_seed,
                                               tcl_temp=0.5, tcl_lambda=1.0,
                                               tcl_hard_ratio=0.5, tcl_adaptive_temp=True):
        # Track total loss for convergence monitoring
        total_loss = 0.0
        total_batches = 0
        
        for c in selected_clients_id:
            # Set deterministic behavior for this client
            client_seed = (trial_seed * 100 + c * 10) % (2**31 - 1)
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(client_seed)

            model = models['clients'][c]
            model.train()
            
            for batch_idx, data in enumerate(dataloaders['train-private'][c]):
                # Use a batch-specific seed for reproducibility
                batch_seed = client_seed + batch_idx
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(batch_seed)
                
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                # Forward pass
                scores, features = model(inputs)
                
                # Make sure features is a tensor, not a list
                if isinstance(features, list):
                    features = features[-1]  # Use the last layer features if it's a list
                
                # Standard cross-entropy loss
                ce_loss = torch.sum(criterion(scores, labels)) / labels.size(0)
                
                # Tuned contrastive loss
                tcl_loss = tuned_contrastive_loss(
                    features, labels, 
                    temperature=tcl_temp, 
                    lambda_weight=tcl_lambda,
                    hard_mining_ratio=tcl_hard_ratio, 
                    adaptive_temp=tcl_adaptive_temp
                )
                
                # Combined loss
                loss = ce_loss + tcl_loss
                
                # Track loss for convergence monitoring
                total_loss += loss.item()
                total_batches += 1
                
                # Debug printout
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        _, pred = torch.max(scores, 1)
                        correct = (pred == labels).sum().item()
                        accuracy = 100 * correct / labels.size(0)
                        print(f"[Debug] Client {c} | Batch {batch_idx} | CE Loss: {ce_loss.item():.4f} | "
                              f"TCL Loss: {tcl_loss.item():.4f} | Total Loss: {loss.item():.4f} | "
                              f"Batch Accuracy: {accuracy:.2f}%")
                
                # Backward and optimize
                self.iters += 1
                optimizers['clients'][c].zero_grad()
                loss.backward()
                optimizers['clients'][c].step()
                
                # Log occasionally
                if (self.iters % 1000 == 0):
                    print(f'Client {c} | Batch {batch_idx} | CE Loss: {ce_loss.item():.4f} | '
                          f'TCL Loss: {tcl_loss.item():.4f} | Total Loss: {loss.item():.4f}')
        
        # Return average loss for this epoch
        avg_loss = total_loss / max(1, total_batches)
        return avg_loss
    
    def _train_epoch_client_simple_contrastive(self, selected_clients_id, models, criterion, 
                                              optimizers, dataloaders, trial_seed,
                                              contrastive_temp=0.5, contrastive_weight=1.0):
        # Track total loss for convergence monitoring
        total_loss = 0.0
        total_batches = 0
        
        for c in selected_clients_id:
            # Set deterministic behavior for this client
            client_seed = (trial_seed * 100 + c * 10) % (2**31 - 1)
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(client_seed)

            model = models['clients'][c]
            model.train()
            
            for batch_idx, data in enumerate(dataloaders['train-private'][c]):
                # Use a batch-specific seed for reproducibility
                batch_seed = client_seed + batch_idx
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(batch_seed)
                
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                # Forward pass
                scores, features = model(inputs)
                
                # Make sure features is a tensor, not a list
                if isinstance(features, list):
                    features = features[-1]  # Use the last layer features if it's a list
                
                # Standard cross-entropy loss
                ce_loss = torch.sum(criterion(scores, labels)) / labels.size(0)
                
                # Simple contrastive loss
                cl_loss = simple_contrastive_loss(
                    features, labels, temperature=contrastive_temp
                )
                
                # Combined loss
                loss = ce_loss + contrastive_weight * cl_loss
                
                # Track loss for convergence monitoring
                total_loss += loss.item()
                total_batches += 1
                
                # Debug printout
                if batch_idx % 100 == 0:
                    with torch.no_grad():
                        _, pred = torch.max(scores, 1)
                        correct = (pred == labels).sum().item()
                        accuracy = 100 * correct / labels.size(0)
                        print(f"[Debug] Client {c} | Batch {batch_idx} | CE Loss: {ce_loss.item():.4f} | "
                              f"CL Loss: {cl_loss.item():.4f} | Total Loss: {loss.item():.4f} | "
                              f"Batch Accuracy: {accuracy:.2f}%")
                
                # Backward and optimize
                self.iters += 1
                optimizers['clients'][c].zero_grad()
                loss.backward()
                optimizers['clients'][c].step()
                
                # Log occasionally
                if (self.iters % 1000 == 0):
                    print(f'Client {c} | Batch {batch_idx} | CE Loss: {ce_loss.item():.4f} | '
                          f'CL Loss: {cl_loss.item():.4f} | Total Loss: {loss.item():.4f}')
        
        # Return average loss for this epoch
        avg_loss = total_loss / max(1, total_batches)
        return avg_loss
    
    def _train_epoch_client_distil(self, selected_clients_id, models, criterion, 
                                  optimizers, dataloaders, comm, trial_seed):
        """
        Train client models with knowledge distillation from the server model.
        
        Args:
            selected_clients_id (list): IDs of selected clients for this round
            models (dict): Dictionary of models
            criterion (nn.Module): Loss function
            optimizers (dict): Dictionary of optimizers
            dataloaders (dict): Dictionary of dataloaders
            comm (int): Current communication round
            trial_seed (int): Seed for reproducibility
            
        Returns:
            float: Average loss for this epoch
        """
        kld = torch.nn.KLDivLoss(reduce=False)
        
        # Track total loss for convergence monitoring
        total_loss = 0.0
        total_batches = 0

        for c in selected_clients_id:
            # Set client-specific seed for reproducibility
            client_seed = (trial_seed * 100 + c * 10) % (2**31 - 1)
            torch.manual_seed(client_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(client_seed)
            
            model = models['clients'][c]
            model.train()
            unlab_data_iterator = self._read_data(dataloaders['unlab-private'][c])

            for batch_idx, data in enumerate(dataloaders['train-private'][c]):
                # Set batch-specific seeds
                batch_seed = client_seed + batch_idx
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(batch_seed)

                # Get labeled data
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                # Get unlabeled data
                unlab_data = next(unlab_data_iterator)
                unlab_inputs = unlab_data[0].to(self.device)

                # Deterministic beta sampling
                m = Beta(torch.FloatTensor([self.config.BETA[0]]).item(), 
                         torch.FloatTensor([self.config.BETA[1]]).item())
                beta_0 = m.sample(sample_shape=torch.Size([unlab_inputs.size(0)])).to(self.device)
                beta = beta_0.view(unlab_inputs.size(0), 1, 1, 1)

                # Deterministic index selection
                batch_rng = np.random.RandomState(batch_seed)
                indices = batch_rng.choice(unlab_inputs.size(0), size=unlab_inputs.size(0), replace=False)
                
                # Create mixed inputs
                mixed_inputs = beta * unlab_inputs + (1 - beta) * unlab_inputs[indices,...]

                # Forward pass on labeled data
                scores, _ = model(inputs)

                self.iters += 1
                optimizers['clients'][c].zero_grad()

                # Get server predictions on mixed unlabeled data
                with torch.no_grad():
                    scores_unlab_t, _ = models['server'](mixed_inputs)

                # Get client predictions on mixed unlabeled data
                scores_unlab, _ = model(mixed_inputs)
                _, pred_labels = torch.max(scores_unlab_t.data, 1)

                # Calculate weight matrix Γ for specialized knowledge
                loss_weight = self.loss_weight_list[c]
                mask = (loss_weight > 0).float()
                weight_ratios = loss_weight.sum() / (loss_weight + 1e-6)
                weight_ratios *= mask
                weights = (beta_0 * (weight_ratios / weight_ratios.sum())[pred_labels] + 
                          (1-beta_0) * (weight_ratios / weight_ratios.sum())[pred_labels[indices]])

                # Compensatory knowledge distillation loss (only after first round)
                distil_loss = (int(comm > 0) * 
                              (weights * kld(F.log_softmax(scores_unlab, -1), 
                                           F.softmax(scores_unlab_t.detach(), -1)).mean(1)).mean())

                # Apply class specialization weights
                spc = loss_weight
                spc = spc.unsqueeze(0).expand(labels.size(0), -1)
                scores = scores + spc.log()

                # Calculate client loss
                loss = torch.sum(criterion(scores, labels)) / labels.size(0)
                
                # Combined KCFU loss
                total_client_loss = loss + distil_loss
                
                # Track loss for convergence monitoring
                total_loss += total_client_loss.item()
                total_batches += 1

                # Backward pass
                total_client_loss.backward()
                optimizers['clients'][c].step()

                # Log occasionally
                if (self.iters % 1000 == 0):
                    print(f'Client {c} | Batch {batch_idx} | ' 
                          f'Loss: {loss.item():.4f} | Distil loss: {distil_loss.item():.4f}')
        
        # Return average loss for this epoch
        avg_loss = total_loss / max(1, total_batches)
        return avg_loss

    def evaluate(self, model, dataloader, mode='test'):
        """
        Evaluate model accuracy on a dataset.
        
        Args:
            model (nn.Module): Model to evaluate
            dataloader (DataLoader): DataLoader for evaluation
            mode (str): Evaluation mode ('test' or 'val')
            
        Returns:
            float: Accuracy percentage
        """
        model.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                scores, _ = model(inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        return 100 * correct / total
    
    def _aggregate_models(self, models, selected_clients_id, data_num):
        """
        Aggregate client models to update the server model using weighted averaging.
        
        Args:
            models (dict): Dictionary of models
            selected_clients_id (list): IDs of selected clients
            data_num (np.ndarray): Number of samples per client
        """
        # Get state dictionaries from client models
        local_states = [copy.deepcopy(models['clients'][c].state_dict()) 
                       for c in selected_clients_id]

        # Get number of samples for selected clients
        selected_data_num = data_num[selected_clients_id]
        
        # Initialize aggregated model with first client's weights
        model_state = local_states[0]

        # Weighted aggregation
        for key in local_states[0]:
            model_state[key] = model_state[key] * selected_data_num[0]
            
            # Add weighted parameters from other clients
            for i in range(1, len(selected_clients_id)):
                model_state[key] = model_state[key].float() + local_states[i][key].float() * selected_data_num[i]
            
            # Normalize by total number of samples
            model_state[key] = model_state[key].float() / np.sum(selected_data_num)
        
        # Update server model
        models['server'].load_state_dict(model_state, strict=False)
    
    def _read_data(self, dataloader):
        """
        Create an infinite data iterator from a dataloader.
        
        Args:
            dataloader (DataLoader): Source dataloader
            
        Returns:
            generator: Infinite data iterator
        """
        while True:
            for data in dataloader:
                yield data
    
    def set_loss_weights(self, loss_weight_list):
        """
        Set class-specific weights for knowledge distillation.
        
        Args:
            loss_weight_list (list): List of tensors with class weights for each client
        """
        self.loss_weight_list = loss_weight_list
        
    def set_data_num(self, data_num):
        """
        Set the number of samples per client for weighted aggregation.
        
        Args:
            data_num (np.ndarray): Number of samples per client
        """
        self.data_num = data_num