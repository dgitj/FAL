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
                # local_log_probs = torch.clamp(local_log_probs, min=-100)
                local_log_probs[local_log_probs == float("-inf")] = 0
                local_log_probs[local_log_probs == float("inf")] = 0                
                local_probabilities = torch.exp(local_log_probs)
                batch_local_entropy = -torch.sum(local_probabilities * local_log_probs, dim=1)
                
                # Calculate global entropy
                global_log_probs = F.log_softmax(global_outputs, dim=1)
                # global_log_probs = torch.clamp(global_log_probs, min=-100)    
                global_log_probs[global_log_probs == float("-inf")] = 0
                global_log_probs[global_log_probs == float("inf")] = 0             
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
    

    def compute_entropy_compatible(self, model, unlabeled_loader, unlabeled_set):
        """
        Computes entropy in the same way as the EntropySampler class for compatibility
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
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Calculate entropy using log_softmax for numerical stability
                log_probs = F.log_softmax(outputs, dim=1)

                # Handle extreme values exactly as in entropy.py
                log_probs[log_probs == float("-inf")] = 0
                log_probs[log_probs == float("inf")] = 0
                
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
    
    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, 
                       num_samples, labeled_set=None, seed=None, global_class_distribution=None, 
                       class_variance_stats=None, current_round=0, total_rounds=5, labeled_set_classes=None):
        """
        Selects samples using a hybrid approach with class-differentiated uncertainty:
        - First selects top 30% samples with highest entropy regardless of class
        - For the two lowest-variance classes: Uses combined local-global uncertainty
        - For high-variance classes: Uses only local entropy
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

        # For the early rounds, use exact entropy.py implementation
        if current_round < 2:
            print(f"Round {current_round}: Using exact entropy.py implementation")
            
            # Implementation copied directly from entropy.py
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
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Calculate entropy using log_softmax for numerical stability
                    log_probs = F.log_softmax(outputs, dim=1)
                    
                    # Handling of extreme values exactly as in entropy.py
                    log_probs[log_probs == float("-inf")] = 0
                    log_probs[log_probs == float("inf")] = 0
                    
                    probabilities = torch.exp(log_probs)
                    batch_entropy = -torch.sum(probabilities * log_probs, dim=1)
                    
                    # Calculate the batch size for this batch
                    batch_size = len(batch_entropy)
                    
                    # Store entropy scores at the correct positions
                    entropy_scores[processed_count:processed_count + batch_size] = batch_entropy.cpu().numpy()
                    processed_count += batch_size
            
            if processed_count != len(unlabeled_set):
                print(f"Warning: Processed {processed_count} samples but unlabeled set size is {len(unlabeled_set)}")
                
            # Debug output
            print(f"Entropy stats - Min: {np.min(entropy_scores)}, Max: {np.max(entropy_scores)}, Mean: {np.mean(entropy_scores)}")
            
            # Sort by entropy in descending order (highest entropy first)
            sorted_indices = np.argsort(-entropy_scores)
            
            # Debug: Print some top and bottom entropy values
            top5_indices = sorted_indices[:5]
            top5_entropy = entropy_scores[top5_indices]
            print(f"Top 5 entropy values: {top5_entropy}")
            
            bottom5_indices = sorted_indices[-5:]
            bottom5_entropy = entropy_scores[bottom5_indices]
            print(f"Bottom 5 entropy values: {bottom5_entropy}")
            
            # Select top samples with highest entropy
            selected_indices = sorted_indices[:num_samples]
            selected_samples = [unlabeled_set[idx] for idx in selected_indices]
            
            # Update remaining unlabeled set
            remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
            remaining_unlabeled = [unlabeled_set[i] for i in remaining_indices]
            
            print(f"Selected {len(selected_samples)} samples using exact entropy.py implementation")
            return selected_samples, remaining_unlabeled
            # Only print this for round 3 and higher
        print(f"Round {current_round}: Using sophisticated hybrid sampling strategy")
        
        # Check if we have the necessary class variance statistics
        if class_variance_stats is None or 'class_stats' not in class_variance_stats:
            raise ValueError("Required class variance statistics not provided. Cannot proceed with hybrid strategy.")
        
        # Find the two classes with lowest variance
        class_variances = [(int(cls), stats['variance']) for cls, stats in class_variance_stats['class_stats'].items()]
        sorted_classes = sorted(class_variances, key=lambda x: x[1])  # Sort by variance (lowest first)
        
        if len(sorted_classes) < 2:
            raise ValueError("Need at least two classes for class-differentiated uncertainty.")
            
        # Get the two lowest-variance classes
        low_var_class1 = sorted_classes[0][0]
        low_var_class2 = sorted_classes[1][0]
        
        print(f"Two classes with lowest variance: Class {low_var_class1} ({sorted_classes[0][1]:.6f}) and Class {low_var_class2} ({sorted_classes[1][1]:.6f})")
        
        # Get class ratios for low-variance classes if available
        low_var_class_ratios = {}
        if global_class_distribution:
            for cls in [low_var_class1, low_var_class2]:
                if str(cls) in global_class_distribution:
                    low_var_class_ratios[cls] = global_class_distribution[str(cls)]
                    print(f"Low variance class {cls} has global ratio: {low_var_class_ratios[cls]:.4f}")
        
        # Calculate current class distribution if we have labeled set classes
        current_distribution = self.calculate_current_distribution(labeled_set, labeled_set_classes)
        if current_distribution:
            print("Current class distribution in labeled set:")
            for cls, proportion in current_distribution.items():
                print(f"  Class {cls}: {proportion:.4f}")
        
        # Compute uncertainty scores from both models
        combined_entropy, local_entropy, global_entropy, predicted_classes, local_probs, global_probs = \
            self.compute_combined_uncertainty(model, model_server, unlabeled_loader, unlabeled_set)
        
        if local_probs is None or global_probs is None:
            raise ValueError("Failed to collect probability data from model predictions.")
        
        # Create masks for different classes
        low_var_mask1 = (predicted_classes == low_var_class1)
        low_var_mask2 = (predicted_classes == low_var_class2)
        low_var_mask = low_var_mask1 | low_var_mask2  # Union of both masks
        high_var_mask = ~low_var_mask  # All other classes
        
        # FIRST PHASE: Select top 30% highest entropy samples regardless of class
        # For first phase selection, use local entropy for all classes to maintain consistency
        pure_entropy_budget = int(num_samples * 0.3)  # 30% for pure entropy selection
        balanced_budget = num_samples - pure_entropy_budget  # 70% for balanced selection
        
        print(f"Entropy-first approach: {pure_entropy_budget} samples by pure entropy, {balanced_budget} with balancing")
        
        # Get all samples sorted by local entropy (highest first) for first phase
        all_entropy_indices = np.argsort(-local_entropy)
        
        # Select top samples by pure entropy
        selected_indices = all_entropy_indices[:pure_entropy_budget].tolist()
        
        # Get count of samples already selected from each category
        selected_low_var1 = sum([1 for idx in selected_indices if low_var_mask1[idx]])
        selected_low_var2 = sum([1 for idx in selected_indices if low_var_mask2[idx]])
        selected_high_var = sum([1 for idx in selected_indices if high_var_mask[idx]])
        
        print(f"First phase selected {selected_low_var1} from first low-variance class, {selected_low_var2} from second low-variance class, "
              f"and {selected_high_var} from high-variance classes based on entropy")
        
        # SECOND PHASE: Calculate the sample allocation for remaining budget based on global distribution
        # but adjusted for what's already been selected
        if balanced_budget > 0:
            # Get remaining samples not yet selected
            remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
            remaining_low_var1 = [i for i in remaining_indices if low_var_mask1[i]]
            remaining_low_var2 = [i for i in remaining_indices if low_var_mask2[i]]
            remaining_high_var = [i for i in remaining_indices if high_var_mask[i]]
            
            # Apply gentler balancing for the second phase
            if global_class_distribution:
                # Get target proportions for each class
                class_targets = {int(cls): prop for cls, prop in global_class_distribution.items()}
                total_selected = {}
                
                # Count already selected samples by class
                for idx in selected_indices:
                    cls = predicted_classes[idx]
                    if cls not in total_selected:
                        total_selected[cls] = 0
                    total_selected[cls] += 1
                
                # Calculate how many samples to select from each class
                needed_by_class = {}
                for cls, target_prop in class_targets.items():
                    # Expected total samples for this class
                    expected = int(num_samples * target_prop)
                    # Already selected samples
                    already_selected = total_selected.get(cls, 0)
                    # Remaining needed
                    needed = max(0, expected - already_selected)
                    needed_by_class[cls] = needed
                    
                print("Samples needed by class for second phase:")
                for cls, needed in needed_by_class.items():
                    print(f"  Class {cls}: {needed} samples")
                
                # Now select samples for each class using appropriate uncertainty metric
                # First, low variance classes using combined uncertainty
                for low_var_class in [low_var_class1, low_var_class2]:
                    needed = needed_by_class.get(low_var_class, 0)
                    if needed > 0:
                        remaining_samples = [i for i in remaining_indices if predicted_classes[i] == low_var_class]
                        
                        if remaining_samples:
                            # Use combined uncertainty for low variance classes
                            uncertainty_scores = combined_entropy[remaining_samples]
                            
                            # Sort by uncertainty, select top needed samples
                            sorted_idx = np.argsort(-uncertainty_scores)  # Descending order
                            selected_cnt = min(needed, len(sorted_idx))
                            
                            # Add selected samples to our list
                            for i in range(selected_cnt):
                                selected_indices.append(remaining_samples[sorted_idx[i]])
                                # Remove from remaining indices
                                remaining_indices.remove(remaining_samples[sorted_idx[i]])
                            
                            print(f"Selected {selected_cnt} samples from low-variance class {low_var_class} using combined uncertainty")
                
                # Now handle high variance classes using local entropy
                for cls in class_targets.keys():
                    if cls not in [low_var_class1, low_var_class2]:  # Only high variance classes
                        needed = needed_by_class.get(cls, 0)
                        if needed > 0:
                            remaining_samples = [i for i in remaining_indices if predicted_classes[i] == cls]
                            
                            if remaining_samples:
                                # Use local entropy for high variance classes
                                uncertainty_scores = local_entropy[remaining_samples]
                                
                                # Sort by uncertainty, select top needed samples
                                sorted_idx = np.argsort(-uncertainty_scores)  # Descending order
                                selected_cnt = min(needed, len(sorted_idx))
                                
                                # Add selected samples to our list
                                for i in range(selected_cnt):
                                    selected_indices.append(remaining_samples[sorted_idx[i]])
                                    # Remove from remaining indices
                                    remaining_indices.remove(remaining_samples[sorted_idx[i]])
                                
                                print(f"Selected {selected_cnt} samples from high-variance class {cls} using local entropy")
            else:
                # Without global distribution, use a simple proportion based on what's remaining
                remaining_low_var1_prop = len(remaining_low_var1) / len(remaining_indices) if remaining_indices else 0
                remaining_low_var2_prop = len(remaining_low_var2) / len(remaining_indices) if remaining_indices else 0
                remaining_high_var_prop = len(remaining_high_var) / len(remaining_indices) if remaining_indices else 0
                
                needed_low_var1 = int(balanced_budget * remaining_low_var1_prop)
                needed_low_var2 = int(balanced_budget * remaining_low_var2_prop)
                needed_high_var = balanced_budget - needed_low_var1 - needed_low_var2
                
                print(f"Second phase needs {needed_low_var1} samples from first low-variance class, "
                      f"{needed_low_var2} from second low-variance class, and {needed_high_var} from high-variance classes")
                
                # Select remaining samples from first low-variance class using combined uncertainty
                if needed_low_var1 > 0 and remaining_low_var1:
                    # Use combined entropy for low variance class
                    low_var_entropy = combined_entropy[remaining_low_var1]
                    
                    # Sort by entropy in descending order (highest combined entropy first)
                    sorted_entropy_indices = np.argsort(-low_var_entropy)
                    
                    # Select samples with highest combined entropy
                    for i in range(min(needed_low_var1, len(sorted_entropy_indices))):
                        idx = remaining_low_var1[sorted_entropy_indices[i]]
                        selected_indices.append(idx)
                        # Remove from remaining indices list to avoid reselection
                        if idx in remaining_indices:
                            remaining_indices.remove(idx)
                
                # Select remaining samples from second low-variance class using combined uncertainty
                if needed_low_var2 > 0 and remaining_low_var2:
                    # Use combined entropy for low variance class
                    low_var_entropy = combined_entropy[remaining_low_var2]
                    
                    # Sort by entropy in descending order (highest combined entropy first)
                    sorted_entropy_indices = np.argsort(-low_var_entropy)
                    
                    # Select samples with highest combined entropy
                    for i in range(min(needed_low_var2, len(sorted_entropy_indices))):
                        idx = remaining_low_var2[sorted_entropy_indices[i]]
                        selected_indices.append(idx)
                        # Remove from remaining indices list to avoid reselection
                        if idx in remaining_indices:
                            remaining_indices.remove(idx)
                
                # Select from high-variance classes using local entropy only
                if needed_high_var > 0 and remaining_high_var:
                    # Use local entropy for high variance classes
                    high_var_entropy = local_entropy[remaining_high_var]
                    
                    # Sort by entropy in descending order (highest local entropy first)
                    sorted_entropy_indices = np.argsort(-high_var_entropy)
                    
                    # Select samples with highest local entropy
                    for i in range(min(needed_high_var, len(sorted_entropy_indices))):
                        idx = remaining_high_var[sorted_entropy_indices[i]]
                        selected_indices.append(idx)
                        # Remove from remaining indices list to avoid reselection
                        if idx in remaining_indices:
                            remaining_indices.remove(idx)
        
        # If we still don't have enough samples (edge case), take more by entropy
        if len(selected_indices) < num_samples:
            remaining_to_select = num_samples - len(selected_indices)
            print(f"Still need {remaining_to_select} more samples, selecting by pure entropy")
            
            # Get indices of samples not already selected
            remaining_indices = [i for i in range(len(unlabeled_set)) if i not in selected_indices]
            
            if not remaining_indices:
                raise ValueError(f"Unable to select the requested {num_samples} samples. Only {len(selected_indices)} available.")
                
            # Sort remaining by appropriate entropy (combined for low variance, local for high variance)
            # We'll use local entropy for this final selection for consistency
            remaining_entropy = local_entropy[remaining_indices]
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
            
        # Analyze the uncertainty metrics used for the selected samples
        low_var1_indices = [idx for idx in selected_indices if predicted_classes[idx] == low_var_class1]
        low_var2_indices = [idx for idx in selected_indices if predicted_classes[idx] == low_var_class2]
        high_var_indices = [idx for idx in selected_indices if idx not in low_var1_indices and idx not in low_var2_indices]
        
        if low_var1_indices:
            local_ent = np.mean(local_entropy[low_var1_indices])
            global_ent = np.mean(global_entropy[low_var1_indices])
            combined_ent = np.mean(combined_entropy[low_var1_indices])
            print(f"Class {low_var_class1} (low var) - Avg Local: {local_ent:.4f}, Avg Global: {global_ent:.4f}, Avg Combined: {combined_ent:.4f}")
            
        if low_var2_indices:
            local_ent = np.mean(local_entropy[low_var2_indices])
            global_ent = np.mean(global_entropy[low_var2_indices])
            combined_ent = np.mean(combined_entropy[low_var2_indices])
            print(f"Class {low_var_class2} (low var) - Avg Local: {local_ent:.4f}, Avg Global: {global_ent:.4f}, Avg Combined: {combined_ent:.4f}")
            
        if high_var_indices:
            local_ent = np.mean(local_entropy[high_var_indices])
            global_ent = np.mean(global_entropy[high_var_indices])
            combined_ent = np.mean(combined_entropy[high_var_indices])
            print(f"High variance classes - Avg Local: {local_ent:.4f}, Avg Global: {global_ent:.4f}, Avg Combined: {combined_ent:.4f}")
        
        # Final checks
        if len(selected_samples) != num_samples:
            raise ValueError(f"Selection error: Selected {len(selected_samples)} but requested {num_samples}")
        
        # Ensure no duplicates
        if len(set(selected_samples)) != len(selected_samples):
            raise ValueError("Selection error: Duplicate samples in the selected set")
        
        print(f"Successfully selected {len(selected_samples)} samples using class-differentiated uncertainty approach")
        return selected_samples, remaining_unlabeled