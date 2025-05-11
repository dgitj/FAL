import numpy as np
import torch
import torch.nn.functional as F

class AblationClassUncertaintySampler:
    def __init__(self, device="cuda"):
        """
        Initializes the ablation sampler that applies different uncertainty metrics
        based on class variance, without phased selection or rebalancing.
        
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
                
                # Get predicted classes from local model (for class-specific uncertainty)
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
        combined_entropy = 0.5 * local_entropy + 0.5 * global_entropy

        return combined_entropy, local_entropy, global_entropy, predicted_classes, local_probs, global_probs
    
    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, 
                     num_samples, labeled_set=None, seed=None, global_class_distribution=None, 
                     class_variance_stats=None, current_round=0, total_rounds=5, labeled_set_classes=None):
        """
        Selects samples using only class-specific uncertainty metrics:
        - For the two lowest-variance classes: Uses combined local-global uncertainty
        - For high-variance classes: Uses only local entropy
        
        No phased selection or rebalancing is performed.
        
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
        """
        # Ensure reproducibility if seed is provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Ensure we don't request more samples than available
        num_samples = min(num_samples, len(unlabeled_set))
        
        # Check if we have the necessary class variance statistics
        if class_variance_stats is None or 'class_stats' not in class_variance_stats:
            raise ValueError("Required class variance statistics not provided. Cannot proceed with class-specific strategy.")
        
        # Find the two classes with lowest variance
        class_variances = [(int(cls), stats['variance']) for cls, stats in class_variance_stats['class_stats'].items()]
        sorted_classes = sorted(class_variances, key=lambda x: x[1])  # Sort by variance (lowest first)
        
        if len(sorted_classes) < 2:
            raise ValueError("Need at least two classes for class-differentiated uncertainty.")
            
        # Get the two lowest-variance classes
        low_var_class1 = sorted_classes[0][0]
        low_var_class2 = sorted_classes[1][0]
        
        print(f"Two classes with lowest variance: Class {low_var_class1} ({sorted_classes[0][1]:.6f}) and Class {low_var_class2} ({sorted_classes[1][1]:.6f})")
        
        # Compute uncertainty scores from both models
        combined_entropy, local_entropy, global_entropy, predicted_classes, local_probs, global_probs = \
            self.compute_combined_uncertainty(model, model_server, unlabeled_loader, unlabeled_set)
        
        if local_probs is None or global_probs is None:
            raise ValueError("Failed to collect probability data from model predictions.")
        
        # Create class-specific masks
        low_var_mask = (predicted_classes == low_var_class1) | (predicted_classes == low_var_class2)
        high_var_mask = ~low_var_mask
        
        # Create a unified uncertainty score array
        unified_uncertainty = np.zeros_like(local_entropy)
        
        # Apply class-specific uncertainty metrics
        unified_uncertainty[low_var_mask] = combined_entropy[low_var_mask]  # Combined for low variance classes
        unified_uncertainty[high_var_mask] = local_entropy[high_var_mask]   # Local for high variance classes
        
        # Count samples by class type (for logging)
        low_var_count = np.sum(low_var_mask)
        high_var_count = np.sum(high_var_mask)
        
        print(f"Unlabeled pool composition: {low_var_count} samples from low-variance classes, {high_var_count} from high-variance classes")
        
        # Select top samples directly based on unified uncertainty scores
        sorted_indices = np.argsort(-unified_uncertainty)  # Sort by uncertainty (highest first)
        selected_indices = sorted_indices[:num_samples].tolist()
        
        # For analysis: count selected samples by class type
        selected_low_var = sum(1 for idx in selected_indices if low_var_mask[idx])
        selected_high_var = sum(1 for idx in selected_indices if high_var_mask[idx])
        
        print(f"Selected {selected_low_var} samples from low-variance classes and {selected_high_var} from high-variance classes")
        
        # Analysis of selected samples by class
        selected_class_distribution = {}
        for idx in selected_indices:
            cls = predicted_classes[idx]
            if str(cls) not in selected_class_distribution:
                selected_class_distribution[str(cls)] = 0
            selected_class_distribution[str(cls)] += 1
        
        print("Selected samples class distribution:")
        for cls, count in selected_class_distribution.items():
            proportion = count / len(selected_indices)
            print(f"  Class {cls}: {count} samples ({proportion:.4f})")
        
        # Analyze the uncertainty metrics used for the selected samples
        low_var_indices = [idx for idx in selected_indices if low_var_mask[idx]]
        high_var_indices = [idx for idx in selected_indices if high_var_mask[idx]]
        
        if low_var_indices:
            local_ent = np.mean(local_entropy[low_var_indices])
            global_ent = np.mean(global_entropy[low_var_indices])
            combined_ent = np.mean(combined_entropy[low_var_indices])
            print(f"Low-variance classes - Avg Local: {local_ent:.4f}, Avg Global: {global_ent:.4f}, Avg Combined: {combined_ent:.4f}")
            
        if high_var_indices:
            local_ent = np.mean(local_entropy[high_var_indices])
            global_ent = np.mean(global_entropy[high_var_indices])
            combined_ent = np.mean(combined_entropy[high_var_indices])
            print(f"High-variance classes - Avg Local: {local_ent:.4f}, Avg Global: {global_ent:.4f}, Avg Combined: {combined_ent:.4f}")
        
        # Convert indices to actual sample IDs
        selected_samples = [unlabeled_set[idx] for idx in selected_indices]
        
        # Update remaining unlabeled set
        remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
        remaining_unlabeled = [unlabeled_set[i] for i in remaining_indices]
        
        # Final checks
        if len(selected_samples) != num_samples:
            raise ValueError(f"Selection error: Selected {len(selected_samples)} but requested {num_samples}")
        
        # Ensure no duplicates
        if len(set(selected_samples)) != len(selected_samples):
            raise ValueError("Selection error: Duplicate samples in the selected set")
        
        print(f"Successfully selected {len(selected_samples)} samples using class-specific uncertainty metrics")
        return selected_samples, remaining_unlabeled