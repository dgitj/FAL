import numpy as np
import torch
import torch.nn.functional as F

class HybridEntropyKAFALSampler:
    def __init__(self, device="cuda"):
        """
        Initializes the HybridEntropyKAFALSampler that combines entropy-based sampling with
        a simplified KAFAL approach for the class with lowest variance.
        
        Args:
            device (str): Device to run the calculations on (e.g., 'cuda' or 'cpu').
        """
        self.device = device
        
    def compute_model_predictions(self, local_model, global_model, unlabeled_loader, unlabeled_set):
        """
        Computes predictions from both local and global models.
        
        Args:
            local_model (torch.nn.Module): The client's local model
            global_model (torch.nn.Module): The global server model
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data
            unlabeled_set (list): The actual indices of unlabeled samples
            
        Returns:
            tuple: (local_entropy, local_probs, global_probs, predicted_classes) 
        """
        local_model.eval()
        global_model.eval()
        
        local_entropy = np.zeros(len(unlabeled_set))
        local_probs_list = []
        global_probs_list = []
        predicted_classes = np.zeros(len(unlabeled_set), dtype=np.int64)
        processed_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass on local model
                local_outputs = local_model(inputs)
                if isinstance(local_outputs, tuple):
                    local_outputs = local_outputs[0]
                
                # Forward pass on global model
                global_outputs = global_model(inputs)
                if isinstance(global_outputs, tuple):
                    global_outputs = global_outputs[0]
                
                # Calculate local entropy using log_softmax for numerical stability
                local_log_probs = F.log_softmax(local_outputs, dim=1)
                local_log_probs = torch.clamp(local_log_probs, min=-100)
                
                local_probabilities = torch.exp(local_log_probs)
                batch_entropy = -torch.sum(local_probabilities * local_log_probs, dim=1)
                
                # Get predicted classes from local model
                _, predicted = torch.max(local_outputs, 1)
                
                # Calculate softmax probabilities for both models
                global_probabilities = F.softmax(global_outputs, dim=1)
                
                # Store softmax probabilities from both models
                local_probs_list.append(local_probabilities.cpu())
                global_probs_list.append(global_probabilities.cpu())
                
                # Calculate batch size
                batch_size = len(batch_entropy)
                
                # Store results
                local_entropy[processed_count:processed_count + batch_size] = batch_entropy.cpu().numpy()
                predicted_classes[processed_count:processed_count + batch_size] = predicted.cpu().numpy()
                processed_count += batch_size
        
        if processed_count != len(unlabeled_set):
            print(f"Warning: Processed {processed_count} samples but unlabeled set size is {len(unlabeled_set)}")
            
        # Concatenate all results
        local_probs = torch.cat(local_probs_list, dim=0) if local_probs_list else None
        global_probs = torch.cat(global_probs_list, dim=0) if global_probs_list else None

        return local_entropy, local_probs, global_probs, predicted_classes
    
    def compute_discrepancy_score(self, local_probs, global_probs, class_idx, class_ratio=None):
        """
        Compute a simplified discrepancy score for the specified class.
        
        Args:
            local_probs (torch.Tensor): Local model softmax probabilities
            global_probs (torch.Tensor): Global model softmax probabilities
            class_idx (int): Class index to focus on
            class_ratio (float, optional): Ratio of this class in global distribution
            
        Returns:
            torch.Tensor: Discrepancy scores
        """
        # Extract confidence scores for the target class from both models
        local_confidence = local_probs[:, class_idx]
        global_confidence = global_probs[:, class_idx]
        
        # Calculate absolute discrepancy between models
        discrepancy = torch.abs(local_confidence - global_confidence)
        
        # Calculate weighted confidence (high local confidence, high discrepancy)
        # This prioritizes samples where the local model is confident but disagrees with global model
        weighted_scores = local_confidence * discrepancy
        
        # If class ratio is provided, optionally adjust for underrepresented classes
        if class_ratio is not None and class_ratio > 0:
            # Simple inverse weighting - lower representation gets higher weight
            weight_factor = 1.0 / (class_ratio * 10 + 0.1)  # +0.1 to avoid division by zero
            # Apply modest weighting
            weighted_scores = weighted_scores * weight_factor
            
        return weighted_scores
    
    def entropy_first_selection(self, local_entropy, predicted_classes, min_var_class, num_samples, 
                               global_class_distribution=None):
        """
        Simplified selection approach that prioritizes high entropy samples while providing
        gentle class balancing.
        
        Args:
            local_entropy (np.ndarray): Entropy scores for unlabeled samples
            predicted_classes (np.ndarray): Predicted classes for unlabeled samples
            min_var_class (int): Class with lowest variance
            num_samples (int): Total number of samples to select
            global_class_distribution (dict, optional): Global class distribution
            
        Returns:
            tuple: (low_var_indices, other_indices, remaining_indices) 
        """
        # First, identify top entropy samples (30% of budget)
        top_entropy_count = int(num_samples * 0.3)  # Reserve 30% for pure entropy selection
        remaining_count = num_samples - top_entropy_count
        
        # Get indices sorted by entropy (highest first)
        all_indices = np.arange(len(local_entropy))
        entropy_sorted_indices = all_indices[np.argsort(-local_entropy)]
        
        # Select top entropy samples regardless of class
        top_entropy_indices = entropy_sorted_indices[:top_entropy_count]
        print(f"Selected {len(top_entropy_indices)} samples based purely on entropy")
        
        # Create masks for remaining samples
        remaining_indices = np.setdiff1d(all_indices, top_entropy_indices)
        remaining_entropy = local_entropy[remaining_indices]
        remaining_classes = predicted_classes[remaining_indices]
        
        # Identify which of the top entropy samples are from low-variance class and which aren't
        top_entropy_classes = predicted_classes[top_entropy_indices]
        low_var_top_entropy = top_entropy_indices[top_entropy_classes == min_var_class]
        other_top_entropy = top_entropy_indices[top_entropy_classes != min_var_class]
        
        # For remaining samples, apply soft balancing
        # Determine target ratio for low variance class
        if global_class_distribution and str(min_var_class) in global_class_distribution:
            # Get the global proportion for the low-variance class
            low_var_target = global_class_distribution[str(min_var_class)]
        else:
            # Default to equal distribution
            low_var_target = 1.0 / len(np.unique(predicted_classes))
        
        # Calculate how many additional samples we need from low variance class
        # Account for those already selected in top entropy
        low_var_selected = len(low_var_top_entropy)
        low_var_target_count = int(num_samples * low_var_target)
        additional_low_var = max(0, low_var_target_count - low_var_selected)
        additional_low_var = min(additional_low_var, remaining_count)
        additional_other = remaining_count - additional_low_var
        
        print(f"Low variance class target: {low_var_target:.4f}, already have {low_var_selected}, need {additional_low_var} more")
        
        # Get remaining indices for each group
        remaining_low_var = remaining_indices[remaining_classes == min_var_class]
        remaining_other = remaining_indices[remaining_classes != min_var_class]
        
        # Sort remaining indices by entropy within each group
        if len(remaining_low_var) > 0:
            low_var_entropy = local_entropy[remaining_low_var]
            sorted_low_var = remaining_low_var[np.argsort(-low_var_entropy)]
            additional_low_var = min(additional_low_var, len(sorted_low_var))
            low_var_indices = np.concatenate([low_var_top_entropy, sorted_low_var[:additional_low_var]])
        else:
            low_var_indices = low_var_top_entropy
        
        if len(remaining_other) > 0:
            other_entropy = local_entropy[remaining_other]
            sorted_other = remaining_other[np.argsort(-other_entropy)]
            additional_other = min(additional_other, len(sorted_other))
            other_indices = np.concatenate([other_top_entropy, sorted_other[:additional_other]])
        else:
            other_indices = other_top_entropy
        
        # Update remaining indices
        final_selected = np.concatenate([low_var_indices, other_indices])
        remaining_all = np.setdiff1d(all_indices, final_selected)
        
        return low_var_indices, other_indices, remaining_all
    
    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, 
                       num_samples, labeled_set=None, seed=None, global_class_distribution=None, 
                       class_variance_stats=None, current_round=0, total_rounds=5, labeled_set_classes=None):
        """
        Selects samples using a hybrid approach with entropy-first selection:
        - First selects top entropy samples regardless of class
        - Then applies soft balancing for remaining selection budget
        - For the class with lowest variance: Uses model discrepancy approach
        - For all other classes: Uses entropy-based sampling
        
        Args:
            model (torch.nn.Module): Client model used for predictions
            model_server (torch.nn.Module): Server model used for discrepancy calculation
            unlabeled_loader (DataLoader): DataLoader for unlabeled data
            client_id (int): ID of the current client
            unlabeled_set (list): List of unlabeled sample indices
            num_samples (int): Number of samples to select
            labeled_set (list, optional): List of labeled sample indices
            seed (int, optional): Random seed for reproducibility
            global_class_distribution (dict, optional): Global class distribution
            class_variance_stats (dict, optional): Statistics about class distribution variance
            current_round (int, optional): Current active learning round
            total_rounds (int, optional): Total number of active learning rounds
            labeled_set_classes (np.ndarray, optional): Classes of samples in the labeled set
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
            
        Raises:
            ValueError: If required data is missing or there are processing errors
        """
        # Ensure reproducibility if seed is provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Ensure we don't request more samples than available
        num_samples = min(num_samples, len(unlabeled_set))
        
        # Check if we have the necessary class variance statistics
        if class_variance_stats is None or 'class_stats' not in class_variance_stats:
            raise ValueError("Required class variance statistics not provided. Cannot proceed with hybrid strategy.")
        
        # Find the class with the lowest variance
        min_var_class = None
        min_variance = float('inf')
        for class_idx, stats in class_variance_stats['class_stats'].items():
            if stats['variance'] < min_variance:
                min_variance = stats['variance']
                min_var_class = int(class_idx)  # Ensure it's an integer
        
        if min_var_class is None:
            raise ValueError("Unable to determine class with lowest variance. Check class variance statistics.")
            
        print(f"Class with lowest variance: {min_var_class} (variance: {min_variance:.6f})")
        
        # Get class ratio for the low-variance class if available
        low_var_class_ratio = None
        if global_class_distribution and str(min_var_class) in global_class_distribution:
            low_var_class_ratio = global_class_distribution[str(min_var_class)]
            print(f"Low variance class {min_var_class} has global ratio: {low_var_class_ratio:.4f}")
        
        # Compute predictions from both models
        local_entropy, local_probs, global_probs, predicted_classes = self.compute_model_predictions(
            model, model_server, unlabeled_loader, unlabeled_set
        )
        
        if local_probs is None or global_probs is None:
            raise ValueError("Failed to collect probability data from model predictions.")
        
        # Use entropy-first selection to determine indices
        low_var_indices, other_indices, remaining_indices = self.entropy_first_selection(
            local_entropy, predicted_classes, min_var_class, num_samples, global_class_distribution
        )
        
        print(f"Entropy-first selection chose {len(low_var_indices)} low-variance class samples")
        print(f"And {len(other_indices)} samples from other classes")
        
        # Process low-variance class indices with discrepancy score
        if len(low_var_indices) > 0:
            # Compute discrepancy scores for low-variance samples
            discrepancy_scores = self.compute_discrepancy_score(
                local_probs[low_var_indices], 
                global_probs[low_var_indices], 
                min_var_class, 
                low_var_class_ratio
            )
            
            # Sort by discrepancy score (highest first)
            sorted_indices = torch.argsort(discrepancy_scores, descending=True)
            sorted_low_var_indices = [low_var_indices[idx.item()] for idx in sorted_indices]
        else:
            sorted_low_var_indices = []
        
        # Other classes are already sorted by entropy in entropy_first_selection
        
        # Combine selected indices
        selected_indices = sorted_low_var_indices + list(other_indices)
        
        # Convert indices to actual sample IDs
        selected_samples = [unlabeled_set[idx] for idx in selected_indices]
        
        # Update remaining unlabeled set
        remaining_unlabeled = [unlabeled_set[idx] for idx in remaining_indices]
        
        # Verification checks
        if len(selected_samples) != num_samples:
            print(f"Warning: Selected {len(selected_samples)} but requested {num_samples}")
            # Handle any discrepancy by selecting more or removing extras
            if len(selected_samples) < num_samples and len(remaining_unlabeled) > 0:
                # Need more samples
                additional_needed = num_samples - len(selected_samples)
                additional_indices = remaining_indices[:additional_needed]
                additional_samples = [unlabeled_set[idx] for idx in additional_indices]
                selected_samples.extend(additional_samples)
                remaining_unlabeled = [idx for idx in remaining_unlabeled if idx not in additional_samples]
            elif len(selected_samples) > num_samples:
                # Too many samples, remove extras
                selected_samples = selected_samples[:num_samples]
        
        # Final verification
        if len(selected_samples) != num_samples:
            print(f"Warning: Final selection has {len(selected_samples)} samples instead of {num_samples}")
        
        # Ensure no duplicates
        selected_samples = list(dict.fromkeys(selected_samples))
        
        print(f"Successfully selected {len(selected_samples)} samples using hybrid entropy-first strategy")
        return selected_samples, remaining_unlabeled