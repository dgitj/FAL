import numpy as np
import torch
import torch.nn.functional as F

class GlobalOptimalEntropyStrategy:
    def __init__(self, device="cuda"):
        """
        Initializes the Global Optimal Entropy strategy that serves as an upper bound baseline.
        This strategy selects samples by considering the entropy scores of all clients
        and allocating the budget globally.
        
        Args:
            device (str): Device to run the calculations on (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        self.client_entropy_scores = {}  # Store entropy scores for each client
        self.budget_allocation = {}      # Store budget allocations for clients
    
    def compute_entropy(self, model, unlabeled_loader, client_id, unlabeled_set):
        """
        Computes the entropy scores for the unlabeled data of a specific client.
        
        Args:
            model (torch.nn.Module): The model used for predictions (global model).
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            client_id (int): ID of the client.
            unlabeled_set (list): List of indices corresponding to the unlabeled data.
        
        Returns:
            numpy.ndarray: Array of entropy scores corresponding to samples in unlabeled_set.
        """
        model.eval()
        entropy_scores = np.zeros(len(unlabeled_set))
        processed_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass with the global model
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take first element if model returns multiple outputs
                
                # Calculate entropy
                probs = F.softmax(outputs, dim=1)
                log_probs = torch.log(probs + 1e-12)  # Add small epsilon for numerical stability
                batch_entropy = -torch.sum(probs * log_probs, dim=1)
                
                # Store entropy scores
                batch_size = len(batch_entropy)
                entropy_scores[processed_count:processed_count + batch_size] = batch_entropy.cpu().numpy()
                processed_count += batch_size
        
        # Store entropy scores for this client
        self.client_entropy_scores[client_id] = {
            'scores': entropy_scores,
            'indices': unlabeled_set
        }
        
        return entropy_scores
    
    def allocate_global_budget(self, clients, total_budget, client_budgets=None):
        """
        Allocates the global budget across clients based on entropy scores.
        
        Args:
            clients (list): List of client IDs.
            total_budget (int): Total sampling budget to allocate.
            client_budgets (dict, optional): Default budget per client if provided.
        
        Returns:
            dict: Dictionary mapping client IDs to their allocated budgets.
        """
        # Log client entropy statistics before allocation
        print("\n===== Client Entropy Statistics =====")
        for c in clients:
            if c in self.client_entropy_scores:
                scores = self.client_entropy_scores[c]['scores']
                if len(scores) > 0:
                    print(f"Client {c}: {len(scores)} samples, "
                          f"Avg entropy: {scores.mean():.4f}, "
                          f"Max entropy: {scores.max():.4f}, "
                          f"Min entropy: {scores.min():.4f}")
                else:
                    print(f"Client {c}: No samples")
        print("==================================\n")
        
        # Collect all scores and indices from all clients
        all_scores = []
        client_indices = []
        
        for c in clients:
            if c in self.client_entropy_scores:
                scores = self.client_entropy_scores[c]['scores']
                indices = self.client_entropy_scores[c]['indices']
                
                # Store scores with client information
                for i, score in enumerate(scores):
                    all_scores.append((c, i, score))
                client_indices.append(len(indices))
        
        # Sort by entropy score (highest first)
        all_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Allocate budget to highest scoring samples
        allocation = {c: 0 for c in clients}
        selected_samples = {c: [] for c in clients}
        
        # Take top samples based on total budget
        for i in range(min(total_budget, len(all_scores))):
            client_id, sample_idx, _ = all_scores[i]
            allocation[client_id] += 1
            # Store the index of the sample in the client's unlabeled set
            sample_original_idx = self.client_entropy_scores[client_id]['indices'][sample_idx]
            selected_samples[client_id].append(sample_original_idx)
        
        # Store the budget allocation and selected samples
        self.budget_allocation = {
            'allocation': allocation,
            'selected_samples': selected_samples
        }
        
        return allocation, selected_samples
    
    def select_samples(self, model, server_model, unlabeled_loader, client_id, unlabeled_set, num_samples, seed=None):
        """
        Selects samples using the Global Optimal Entropy strategy.
        
        Args:
            model (torch.nn.Module): The local client model (not used in this strategy).
            server_model (torch.nn.Module): The global server model used for entropy calculation.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            client_id (int): ID of the client.
            unlabeled_set (list): List of indices corresponding to the unlabeled data.
            num_samples (int): Number of samples that would typically be selected (used as fallback).
            seed (int, optional): Random seed for reproducibility.
        
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        # Calculate entropy scores using the global model
        self.compute_entropy(server_model, unlabeled_loader, client_id, unlabeled_set)
        
        # If this is the first client to call select_samples, we need to wait for other clients
        # Since this is simulated in a single process, we can use the client ID to determine
        # if we need to allocate the budget
        
        # Check if this client is allocated in the budget allocation
        if client_id in self.budget_allocation.get('selected_samples', {}):
            # Use the pre-allocated samples for this client
            selected_samples = self.budget_allocation['selected_samples'][client_id]
        else:
            # Fallback to selecting based on local entropy (should not happen in practice)
            # This would only occur if select_samples is called with a client_id that wasn't
            # part of the budget allocation process
            entropy_scores = self.client_entropy_scores.get(client_id, {}).get('scores', [])
            if len(entropy_scores) > 0:
                # Sort indices by entropy scores
                sorted_indices = np.argsort(-entropy_scores)[:num_samples]
                selected_samples = [unlabeled_set[idx] for idx in sorted_indices]
            else:
                # If no entropy scores available,print error
                print("Error no entropy scores available")
        
        # Compute remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        return selected_samples, remaining_unlabeled