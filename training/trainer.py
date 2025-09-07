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
        
        # Add class distribution tracking
        self.client_class_distributions = {}
        self.global_class_distribution = None
    
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
                else:  # KCFU
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
        
    def aggregate_class_distributions(self):
        """
        Aggregate class distributions from all clients to estimate global distribution.
        
        Returns:
            dict: Global class distribution percentages
        """
        if not self.client_class_distributions:
            return None
            
        # Initialize global counts
        global_counts = {cls: 0 for cls in range(self.config.NUM_CLASSES)}
        
        # Aggregate counts from all clients
        for dist in self.client_class_distributions.values():
            for cls, count in dist.items():
                global_counts[cls] += count
        
        # Calculate percentages
        total_samples = sum(global_counts.values())
        if total_samples > 0:
            self.global_class_distribution = {
                cls: count / total_samples 
                for cls, count in global_counts.items()
            }
            
            print("[Trainer] Global class distribution from all clients:")
            for cls in range(self.config.NUM_CLASSES):
                print(f"  Class {cls}: {self.global_class_distribution[cls]:.4f} ({global_counts[cls]} samples)")
                
            return self.global_class_distribution
        else:
            return None
    
    def update_client_distribution(self, client_id, labeled_set, dataset):
        """
        Calculate and store class distribution for a client.
        
        Args:
            client_id (int): Client identifier
            labeled_set (list): List of indices of labeled samples
            dataset: Dataset containing samples
            
        Returns:
            dict: Class distribution for this client
        """
        class_counts = {cls: 0 for cls in range(self.config.NUM_CLASSES)}
        
        # Count samples per class using ground truth labels
        for idx in labeled_set:
            _, label = dataset[idx]
            class_counts[label] += 1
            
        self.client_class_distributions[client_id] = class_counts
        return class_counts
    
    def get_global_distribution(self):
        """Returns the current global distribution"""
        return self.global_class_distribution
        
    def compute_class_distribution_statistics(self, client_class_distributions, num_classes):
        """
        Compute statistics about class distributions across clients.
        
        Args:
            client_class_distributions (dict): Dictionary mapping client_id to their class distribution
            num_classes (int): Total number of classes
            
        Returns:
            dict: Statistics including mean, variance, and coefficient of variation for each class
        """
        # Initialize arrays to store distributions
        client_count = len(client_class_distributions)
        if client_count == 0:
            return None
        
        # Convert absolute counts to percentages for each client
        normalized_distributions = {}
        for client_id, counts in client_class_distributions.items():
            total = sum(counts.values())
            if total > 0:
                normalized_distributions[client_id] = {
                    cls: counts.get(cls, 0) / total for cls in range(num_classes)
                }
        
        # Create a matrix where each row is a client and each column is a class
        distribution_matrix = np.zeros((client_count, num_classes))
        for i, (client_id, distribution) in enumerate(normalized_distributions.items()):
            for cls in range(num_classes):
                distribution_matrix[i, cls] = distribution.get(cls, 0)
        
        # Compute statistics for each class across clients
        class_stats = {}
        for cls in range(num_classes):
            class_percentages = distribution_matrix[:, cls]
            
            mean = np.mean(class_percentages)
            variance = np.var(class_percentages)
            std_dev = np.std(class_percentages)
            cv = (std_dev / mean) * 100 if mean > 0 else float('inf')  # Coefficient of variation as percentage
            
            class_stats[cls] = {
                'mean': mean,
                'variance': variance,
                'std_dev': std_dev,
                'cv': cv,
                'min': np.min(class_percentages),
                'max': np.max(class_percentages),
                'range': np.max(class_percentages) - np.min(class_percentages)
            }
        
        # Compute overall distribution imbalance metrics
        total_variance = np.sum([stats['variance'] for stats in class_stats.values()])
        avg_cv = np.mean([stats['cv'] for stats in class_stats.values() if not np.isinf(stats['cv'])])
        
        return {
            'class_stats': class_stats,
            'total_variance': total_variance,
            'avg_cv': avg_cv
        }
    
    def analyze_class_distribution_variance(self):
        """
        Analyze the variance of class distributions across clients.
        
        Returns:
            dict: Statistics about class distribution variance
        """
        if not self.client_class_distributions:
            return None
            
        # Call the function to compute statistics
        stats = self.compute_class_distribution_statistics(
            self.client_class_distributions, 
            self.config.NUM_CLASSES
        )
        
        # Print a summary of the findings
        print("\n=== Class Distribution Variance Analysis ===")
        print(f"Total variance across all classes: {stats['total_variance']:.6f}")
        print(f"Average coefficient of variation: {stats['avg_cv']:.2f}%")
        
        # Print per-class statistics
        print("\nPer-class statistics:")
        for cls in range(self.config.NUM_CLASSES):
            cls_stats = stats['class_stats'][cls]
            print(f"  Class {cls}:")
            print(f"    Mean: {cls_stats['mean']*100:.2f}%")
            print(f"    Std Dev: {cls_stats['std_dev']*100:.2f}%")
            print(f"    CV: {cls_stats['cv']:.2f}%")
            print(f"    Range: {cls_stats['range']*100:.2f}% (Min: {cls_stats['min']*100:.2f}%, Max: {cls_stats['max']*100:.2f}%)")
        
        return stats