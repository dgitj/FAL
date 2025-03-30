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
    
    def train(self, models, criterion, optimizers, schedulers, dataloaders, num_epochs, trial_seed):
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
            
        Returns:
            None
        """
        print('>> Train a Model.')

        for comm in range(self.config.COMMUNICATION):
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
            for epoch in range(num_epochs):
                # Choose appropriate training method based on config
                if self.config.LOCAL_MODEL_UPDATE == "Vanilla":
                    self._train_epoch_client_vanilla(
                        selected_clients_id, models, criterion, 
                        optimizers, dataloaders, trial_seed + epoch
                    )
                else:  # KCFU
                    self._train_epoch_client_distil(
                        selected_clients_id, models, criterion, 
                        optimizers, dataloaders, comm, trial_seed + epoch
                    )

                # Update learning rate schedulers
                for c in selected_clients_id:
                    schedulers['clients'][c].step()
                
                print(f'Epoch: {epoch + 1}/{num_epochs} | '
                      f'Communication round: {comm + 1}/{self.config.COMMUNICATION}')    
                      
            end_time = time.time()
            print(f'Average time per epoch: {(end_time-start_time)/num_epochs:.2f} seconds')

            # Log metrics if logger is available
            if self.logger:
                for c in selected_clients_id:
                    self.logger.log_gradient_alignment(
                        0, models['clients'][c], models['server'], 
                        dataloaders['train-private'][c], c
                    )
                    self.logger.log_knowledge_gap(
                        0, models['clients'][c], models['server'], 
                        dataloaders['test'], c
                    )

            # Aggregate client models to update the server model
            self._aggregate_models(models, selected_clients_id, self.data_num)

        print('>> Training completed.')
    
    def _train_epoch_client_vanilla(self, selected_clients_id, models, criterion, 
                                   optimizers, dataloaders, trial_seed):
        """
        Train client models using vanilla federated learning (no knowledge distillation).
        
        Args:
            selected_clients_id (list): IDs of selected clients for this round
            models (dict): Dictionary of models
            criterion (nn.Module): Loss function
            optimizers (dict): Dictionary of optimizers
            dataloaders (dict): Dictionary of dataloaders
            trial_seed (int): Seed for reproducibility
        """
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
                
                # Backward and optimize
                self.iters += 1
                optimizers['clients'][c].zero_grad()
                loss.backward()
                optimizers['clients'][c].step()
                
                # Log occasionally
                if (self.iters % 1000 == 0):
                    print(f'Client {c} | Batch {batch_idx} | Loss: {loss.item():.4f}')
    
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
        """
        kld = torch.nn.KLDivLoss(reduce=False)

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
                (loss + distil_loss).backward()
                optimizers['clients'][c].step()

                # Log occasionally
                if (self.iters % 1000 == 0):
                    print(f'Client {c} | Batch {batch_idx} | ' 
                          f'Loss: {loss.item():.4f} | Distil loss: {distil_loss.item():.4f}')

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