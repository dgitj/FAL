import numpy as np
import torch
import torch.nn.functional as F

class EntropySampler:
    def __init__(self, device="cuda"):
        """
        Initializes the EntropySampler.
        Args:
            device (str): Device to run the calculations on (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        
    def compute_entropy(self, model, unlabeled_loader):
        """
        Computes the entropy of the model predictions for the unlabeled data.
        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
        Returns:
            tuple: Entropy scores and their corresponding indices.
        """
        model.eval()
        entropy_scores = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                    if hasattr(batch[1], 'item'):  # If second element is indices
                        batch_indices = batch[1].tolist()
                    else:
                        batch_indices = list(range(batch_idx * unlabeled_loader.batch_size,
                                               min((batch_idx + 1) * unlabeled_loader.batch_size,
                                                   len(unlabeled_loader.dataset))))
                else:
                    inputs = batch.to(self.device)
                    batch_indices = list(range(batch_idx * unlabeled_loader.batch_size,
                                           min((batch_idx + 1) * unlabeled_loader.batch_size,
                                               len(unlabeled_loader.dataset))))
                
                # Forward pass to get predictions
                try:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take first element if model returns multiple outputs
                except:
                    raise ValueError("Model forward pass failed. Check your model architecture.")
                
                # Calculate entropy using log_softmax for numerical stability
                log_probs = F.log_softmax(outputs, dim=1)
                probabilities = torch.exp(log_probs)
                batch_entropy = -torch.sum(probabilities * log_probs, dim=1)
                
                # Store entropy scores and indices
                entropy_scores.extend(batch_entropy.cpu().numpy())
                indices.extend(batch_indices)
                
        return np.array(entropy_scores), indices
        
    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples):
        """
        Selects the samples with the highest entropy.
        Args:
            model (torch.nn.Module): The model used to generate predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            unlabeled_set (list): List of indices corresponding to the unlabeled data.
            num_samples (int): Number of samples to select.
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        # Ensure we don't request more samples than available
        num_samples = min(num_samples, len(unlabeled_set))

        print(f"Before Selection - Unlabeled Set Size: {len(unlabeled_set)}")
        
        # Get entropy scores and their corresponding dataset indices
        entropy_scores, data_indices = self.compute_entropy(model, unlabeled_loader)
        
        if len(entropy_scores) == 0:
            # Fallback to random selection if computation fails
            selected_indices = np.random.choice(len(unlabeled_set), num_samples, replace=False)
            selected_samples = [unlabeled_set[i] for i in selected_indices]
            remaining_unlabeled = [idx for i, idx in enumerate(unlabeled_set) if i not in selected_indices]
            return selected_samples, remaining_unlabeled
        
        print(f"After Selection (Fallback) - Remaining Unlabeled Set Size: {len(remaining_unlabeled)}")

        # Sort by entropy in descending order (highest entropy first)
        sorted_indices = np.argsort(-entropy_scores)
        
        # Get the top num_samples with highest entropy
        selected_indices = sorted_indices[:num_samples]
        selected_samples = [data_indices[idx] for idx in selected_indices]

        print(f"Selected Indices: {selected_samples[:10]} (showing first 10)")
        
        # Update remaining unlabeled set
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]

        print(f"After Selection - Remaining Unlabeled Set Size: {len(remaining_unlabeled)}")
        
        return selected_samples, remaining_unlabeled