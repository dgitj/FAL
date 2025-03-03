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
        
    def compute_entropy(self, model, unlabeled_loader, unlabeled_set):
        """
        Computes the entropy of the model predictions for the unlabeled data.
        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            unlabeled_set (list): The actual indices of unlabeled samples.
        Returns:
            numpy.ndarray: Array of entropy scores corresponding to samples in unlabeled_set.
        """
        model.eval()
        entropy_scores = np.zeros(len(unlabeled_set))
        processed_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass to get predictions
                try:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take first element if model returns multiple outputs
                except Exception as e:
                    print(f"Error in model forward pass: {e}")
                    raise ValueError("Model forward pass failed. Check your model architecture.")
                
                # Calculate entropy using log_softmax for numerical stability
                log_probs = F.log_softmax(outputs, dim=1)
                probabilities = torch.exp(log_probs)
                batch_entropy = -torch.sum(probabilities * log_probs, dim=1)
                
                # Calculate the batch size for this batch
                batch_size = len(batch_entropy)
                
                # Store entropy scores at the correct positions
                entropy_scores[processed_count:processed_count + batch_size] = batch_entropy.cpu().numpy()
                processed_count += batch_size
        
        if processed_count != len(unlabeled_set):
            print(f"Warning: Processed {processed_count} samples but unlabeled set size is {len(unlabeled_set)}")

        return entropy_scores
        
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

        # print(f"Before Selection - Unlabeled Set Size: {len(unlabeled_set)}")
        
        # Compute entropy scores for each sample in unlabeled_set
        entropy_scores = self.compute_entropy(model, unlabeled_loader, unlabeled_set)
        
        if len(entropy_scores) == 0 or np.var(entropy_scores) < 1e-5:
            print("⚠️ Warning: No meaningful entropy scores. Falling back to random selection.")
            # Fallback to random selection if entropy scores aren't useful
            selected_indices = np.random.choice(len(unlabeled_set), num_samples, replace=False)
            selected_samples = [unlabeled_set[i] for i in selected_indices]
            
            remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
            remaining_unlabeled = [unlabeled_set[i] for i in remaining_indices]
            
            # print(f"After Selection (Fallback) - Remaining Unlabeled Set Size: {len(remaining_unlabeled)}")
            return selected_samples, remaining_unlabeled
        
        # Sort by entropy in descending order (highest entropy first)
        sorted_indices = np.argsort(-entropy_scores)
        
        # Get the top num_samples with highest entropy
        selected_indices = sorted_indices[:num_samples]
        selected_samples = [unlabeled_set[idx] for idx in selected_indices]

        # print(f"Selected Indices: {selected_samples[:5]} (showing first 5)")
        # print(f"Entropy of selected samples: {entropy_scores[selected_indices][:5]}")
        
        # Update remaining unlabeled set
        remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
        remaining_unlabeled = [unlabeled_set[i] for i in remaining_indices]

        # print(f"After Selection - Remaining Unlabeled Set Size: {len(remaining_unlabeled)}")
        
        # Debug: Check for duplicate indices
        if len(set(selected_samples)) != len(selected_samples):
            print("Warning: Duplicate indices in selected samples!")
            
        # Debug: Check if any selected samples remain in the remaining set
        intersection = set(selected_samples).intersection(set(remaining_unlabeled))
        if intersection:
            print(f"Warning: {len(intersection)} selected samples still in remaining set!")
        
        return selected_samples, remaining_unlabeled

