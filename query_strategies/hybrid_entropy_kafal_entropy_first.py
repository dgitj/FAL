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
        
    def compute_combined_uncertainty(self, local_model, global_model, unlabeled_loader, unlabeled_set):
        """
        Computes uncertainty (entropy) from both local and global models and combines them.
        
        Args:
            local_model (torch.nn.Module): The client's local model
            global_model (torch.nn.Module): The global server model
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data
            unlabeled_set (list): The actual indices of unlabeled samples
            
        Returns:
            tuple: (combined_entropy, local_entropy, global_entropy, predicted_classes, local_probs, global_probs) 
        """
        local_model.eval()
        global_model.eval()
        
        local_entropy = np.zeros(len(unlabeled_set))
        global_entropy = np.zeros(len(unlabeled_set))
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
                
                # Calculate local entropy
                local_log_probs = F.log_softmax(local_outputs, dim=1)
                local_log_probs = torch.clamp(local_log_probs, min=-100)                
                local_probabilities = torch.exp(local_log_probs)
                batch_local_entropy = -torch.sum(local_probabilities * local_log_probs, dim=1)
                
                # Calculate global entropy
                global_log_probs = F.log_softmax(global_outputs, dim=1)
                global_log_probs = torch.clamp(global_log_probs, min=-100)                
                global_probabilities = torch.exp(global_log_probs)
                batch_global_entropy = -torch.sum(global_probabilities * global_log_probs, dim=1)
                
                # Get predicted classes from local model (for tracking class distribution)
                _, predicted = torch.max(local_outputs, 1)
                
                # Store softmax probabilities from both models
                local_probs_list.append(local_probabilities.cpu())
                global_probs_list.append(global_probabilities.cpu())
                
                # Calculate batch size
                batch_size = len(batch_local_entropy)
                
                # Store results
                local_entropy[processed_count:processed_count + batch_size] = batch_local_entropy.cpu().numpy()
                global_entropy[processed_count:processed_count + batch_size] = batch_global_entropy.cpu().numpy()
                predicted_classes[processed_count:processed_count + batch_size] = predicted.cpu().numpy()
                processed_count += batch_size
        
        if processed_count != len(unlabeled_set):
            print(f"Warning: Processed {processed_count} samples but unlabeled set size is {len(unlabeled_set)}")
            
        # Concatenate all results
        local_probs = torch.cat(local_probs_list, dim=0) if local_probs_list else None
        global_probs = torch.cat(global_probs_list, dim=0) if global_probs_list else None

        # Combine local and global uncertainty with equal weighting
        # We could adjust these weights if needed (e.g., alpha * local + (1-alpha) * global)
        combined_entropy = 0.5 * local_entropy + 0.5 * global_entropy

        return combined_entropy, local_entropy, global_entropy, predicted_classes, local_probs, global_probs
    
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
    
    def calculate_current_distribution(self, labeled_set=None, labeled_set_classes=None):
        """
        Calculate the current class distribution in the labeled set.
        
        Args:
            labeled_set (list, optional): The current labeled set indices
            labeled_set_classes (np.ndarray, optional): Classes of samples in the labeled set
            
        Returns:
            dict: Dictionary mapping class indices to their proportions
        """
        # If no labeled set info is provided, return empty distribution
        if (labeled_set is None or len(labeled_set) == 0) and \
           (labeled_set_classes is None or len(labeled_set_classes) == 0):
            return {}
            
        # If we have class information directly
        if labeled_set_classes is not None and len(labeled_set_classes) > 0:
            classes = labeled_set_classes
        else:
            # Without class information, we can't calculate distribution
            print("Warning: No class information for labeled set. Cannot calculate distribution.")
            return {}
            
        # Count occurrences of each class
        unique_classes, counts = np.unique(classes, return_counts=True)
        total = len(classes)
        
        # Calculate proportions
        distribution = {}
        for cls, count in zip(unique_classes, counts):
            distribution[str(cls)] = count / total
            
        return distribution
    
    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, 
                       num_samples, labeled_set=None, seed=None, global_class_distribution=None, 
                       class_variance_stats=None, current_round=0, total_rounds=5, labeled_set_classes=None):
        """
        Selects samples using a hybrid approach:
        - First selects top 30% samples with highest combined uncertainty regardless of class
        - Uses combined local-global uncertainty for all classes
        - Applies gentle rebalancing to remaining samples
        
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
        
        # Calculate current class distribution if we have labeled set classes
        current_distribution = self.calculate_current_distribution(labeled_set, labeled_set_classes)
        if current_distribution:
            print("Current class distribution in labeled set:")
            for cls, proportion in current_distribution.items():
                print(f"  Class {cls}: {proportion:.4f}")
        
        # Compute uncertainty scores from both models and combine them
        combined_entropy, local_entropy, global_entropy, predicted_classes, local_probs, global_probs = \
            self.compute_combined_uncertainty(model, model_server, unlabeled_loader, unlabeled_set)
        
        if local_probs is None or global_probs is None:
            raise ValueError("Failed to collect probability data from model predictions.")
        
        # Create masks for the low-variance class and all other classes
        low_var_mask = (predicted_classes == min_var_class)
        other_classes_mask = ~low_var_mask
        
        # FIRST PHASE: Select top 30% highest combined entropy samples regardless of class
        pure_entropy_budget = int(num_samples * 0.3)  # 30% for pure entropy selection
        balanced_budget = num_samples - pure_entropy_budget  # 70% for balanced selection
        
        print(f"Entropy-first approach: {pure_entropy_budget} samples by pure entropy, {balanced_budget} with balancing")
        
        # Get all samples sorted by combined entropy (highest first)
        all_entropy_indices = np.argsort(-combined_entropy)
        
        # Select top samples by pure entropy
        selected_indices = all_entropy_indices[:pure_entropy_budget].tolist()
        
        # Get count of samples already selected from each category
        selected_low_var = sum([1 for idx in selected_indices if low_var_mask[idx]])
        selected_other = sum([1 for idx in selected_indices if other_classes_mask[idx]])
        
        print(f"First phase selected {selected_low_var} low-variance samples and {selected_other} others based on combined entropy")
        
        # SECOND PHASE: Calculate the sample allocation for remaining budget based on global distribution
        # but adjusted for what's already been selected
        if balanced_budget > 0:
            # Get remaining samples not yet selected
            remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
            remaining_low_var = [i for i in remaining_indices if low_var_mask[i]]
            remaining_other = [i for i in remaining_indices if other_classes_mask[i]]
            
            # Apply gentler balancing for the second phase
            if global_class_distribution and str(min_var_class) in global_class_distribution:
                # Target proportion for this class
                target_prop = global_class_distribution[str(min_var_class)]
                
                # Calculate expected number of samples for this class in total selection
                expected_low_var = int(num_samples * target_prop)
                
                # Adjust for already selected samples
                needed_low_var = max(0, expected_low_var - selected_low_var)
                needed_low_var = min(needed_low_var, len(remaining_low_var))  # Can't select more than available
                
                # Remaining samples for other classes
                needed_other = balanced_budget - needed_low_var
            else:
                # Without global distribution, use a simple proportion based on what's remaining
                remaining_low_var_prop = len(remaining_low_var) / (len(remaining_low_var) + len(remaining_other)) if (len(remaining_low_var) + len(remaining_other)) > 0 else 0
                needed_low_var = int(balanced_budget * remaining_low_var_prop)
                needed_other = balanced_budget - needed_low_var
            
            print(f"Second phase needs {needed_low_var} low-variance samples and {needed_other} others")
            
            # Select remaining samples from low-variance class using combined uncertainty (not discrepancy)
            if needed_low_var > 0 and remaining_low_var:
                # Get combined entropy scores for low-variance class
                low_var_entropy = combined_entropy[remaining_low_var]
                
                # Sort by entropy in descending order (highest entropy first)
                sorted_entropy_indices = np.argsort(-low_var_entropy)
                
                # Select indices with highest entropy scores
                for i in range(min(needed_low_var, len(sorted_entropy_indices))):
                    selected_indices.append(remaining_low_var[sorted_entropy_indices[i]])
            
            # Select from other classes using combined entropy
            if needed_other > 0 and remaining_other:
                # Get combined entropy scores for other classes
                other_entropy = combined_entropy[remaining_other]
                
                # Sort by entropy in descending order (highest entropy first)
                sorted_entropy_indices = np.argsort(-other_entropy)
                
                # Add top entropy samples
                for i in range(min(needed_other, len(sorted_entropy_indices))):
                    selected_indices.append(remaining_other[sorted_entropy_indices[i]])
        
        # If we still don't have enough samples (edge case), take more by entropy
        if len(selected_indices) < num_samples:
            remaining_to_select = num_samples - len(selected_indices)
            print(f"Still need {remaining_to_select} more samples, selecting by pure entropy")
            
            # Get indices of samples not already selected
            remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
            
            if not remaining_indices:
                raise ValueError(f"Unable to select the requested {num_samples} samples. Only {len(selected_indices)} available.")
                
            # Sort remaining by combined entropy
            remaining_entropy = combined_entropy[remaining_indices]
            sorted_remaining = np.argsort(-remaining_entropy)
            
            # Add more samples as needed
            for i in range(min(remaining_to_select, len(sorted_remaining))):
                selected_indices.append(remaining_indices[sorted_remaining[i]])
        
        # Convert indices to actual sample IDs
        selected_samples = [unlabeled_set[idx] for idx in selected_indices]
        
        # Update remaining unlabeled set
        remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
        remaining_unlabeled = [unlabeled_set[i] for i in remaining_indices]
        
        # Verify our selection
        final_class_distribution = {}
        for idx in selected_indices:
            cls = predicted_classes[idx]
            if str(cls) not in final_class_distribution:
                final_class_distribution[str(cls)] = 0
            final_class_distribution[str(cls)] += 1
        
        print("Final selection class distribution:")
        for cls, count in final_class_distribution.items():
            proportion = count / len(selected_indices)
            print(f"  Class {cls}: {count} samples ({proportion:.4f})")
            
        # Final checks
        if len(selected_samples) != num_samples:
            raise ValueError(f"Selection error: Selected {len(selected_samples)} but requested {num_samples}")
        
        # Ensure no duplicates
        if len(set(selected_samples)) != len(selected_samples):
            raise ValueError("Selection error: Duplicate samples in the selected set")
        
        print(f"Successfully selected {len(selected_samples)} samples using hybrid entropy-first strategy")
        
        # Output statistics comparing local and global uncertainty
        selected_local_entropy = local_entropy[selected_indices]
        selected_global_entropy = global_entropy[selected_indices]
        print(f"Selected samples - Avg Local Entropy: {np.mean(selected_local_entropy):.4f}, "
              f"Avg Global Entropy: {np.mean(selected_global_entropy):.4f}")
        
        return selected_samples, remaining_unlabeled