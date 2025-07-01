import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
import config  # Import config to access NUM_CLASSES
import numpy as np
import random

class AHFALSampler:
    def __init__(self, device="cuda"):
        """
        Initializes the AHFAL sampler.
        This sampler uses the local model for pseudo-labeling data and balancing class distribution,
        selecting samples with the highest entropy (uncertainty) per pseudo-class.

        Args:
            device (str): Device to run calculations on (e.g., 'cuda' or 'cpu').
        """
        # Check if CUDA is available, fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device
            
        self.debug = True  # Enable detailed debugging
        
        # We'll estimate the global class distribution from labeled data in each cycle
        self.global_class_distribution = None
        
        # For tracking client progress across cycles
        self.client_cycles = {}
        self.client_labeled_sets = {}

    def calculate_local_class_distribution(self, labeled_set, dataset):
        """
        Calculate the class distribution of labeled samples on this client.
        
        Args:
            labeled_set (list): Indices of labeled samples on this client
            dataset: Dataset containing the samples and labels
            
        Returns:
            dict: Counts of samples for each class
        """
        class_counts = {cls: 0 for cls in range(config.NUM_CLASSES)}
        
        # Count samples per class using ground truth labels
        for idx in labeled_set:
            _, label = dataset[idx]
            class_counts[label] += 1
        
        for cls in range(config.NUM_CLASSES):
            percentage = (class_counts.get(cls, 0) / len(labeled_set) * 100) if len(labeled_set) > 0 else 0
            # print(f"  Class {cls}: {class_counts.get(cls, 0)} samples ({percentage:.1f}%)")
            
        return class_counts

    def aggregate_class_distributions(self, client_distributions):
        """
        Aggregate class distributions from all clients.
        
        Args:
            client_distributions (list): List of dictionaries, each containing
                                        class counts from one client
        
        Returns:
            dict: Global class distribution percentages
        
        Raises:
            ValueError: If no labeled samples are available across all clients
        """
        # Initialize global counts
        global_counts = {cls: 0 for cls in range(config.NUM_CLASSES)}
        
        # Aggregate counts from all clients
        for dist in client_distributions:
            for cls, count in dist.items():
                global_counts[cls] += count
        
        # Calculate percentages
        total_samples = sum(global_counts.values())
        if total_samples > 0:
            global_distribution = {cls: count / total_samples 
                                for cls, count in global_counts.items()}
        else:
            # Return error if no labeled samples are available
            raise ValueError("[AHFAL] Error: No labeled samples available for distribution estimation")
        
        
        return global_distribution

    def compute_target_counts(self, current_distribution, num_samples, labeled_set_size, available_classes):
        """
        Compute the target number of samples to select from each class with more aggressive balancing.
        
        Args:
            current_distribution (dict): Current class distribution.
            num_samples (int): Number of samples to select.
            labeled_set_size (int): Current size of the labeled set.
            available_classes (set): Set of classes available in the unlabeled pool.
            
        Returns:
            dict: Target number of samples to select from each class.
        """
        target_counts = {}
        future_size = labeled_set_size + num_samples
        
        # Safety check - if global_class_distribution is not set
        if self.global_class_distribution is None:
            raise ValueError("[AHFAL] Error: Global class distribution not available")
        
        
        # Check if all available classes have the same representation
        available_class_counts = {}
        for cls in available_classes:
            available_class_counts[cls] = current_distribution.get(cls, 0) * labeled_set_size
        
        # Calculate representation ratios for available classes
        representation_ratios = {}
        missing_classes = []
        for cls in available_classes:
            current_count = available_class_counts[cls]
            target_global_ratio = self.global_class_distribution[cls]
            
            # Calculate representation ratio
            if labeled_set_size > 0:
                current_ratio = current_distribution.get(cls, 0)
                ratio = current_ratio / target_global_ratio if target_global_ratio > 0 else float('inf')
                representation_ratios[cls] = ratio
            else:
                # If no labeled samples yet, all classes are equally unrepresented
                representation_ratios[cls] = 0.0
            
            # Identify missing classes (0 samples)
            if current_count == 0:
                missing_classes.append(cls)

        # Determine if we have missing classes to prioritize
        have_missing_classes = len(missing_classes) > 0
        
        # First approach: Prioritize missing classes
        if have_missing_classes and len(missing_classes) <= num_samples:
            # Assign at least one sample to each missing class
            initial_allocation = min(5, num_samples // len(missing_classes))  # More aggressive initial allocation
            for cls in missing_classes:
                target_counts[cls] = initial_allocation
            remaining = num_samples - sum(target_counts.values())
            
            # Distribute remaining samples to all available classes inversely proportional to representation
            if remaining > 0:
                # Prepare for inverse-proportional allocation
                total_inverse_ratio = 0
                inverse_ratios = {}
                
                for cls in available_classes:
                    ratio = representation_ratios.get(cls, 0)
                    inverse = 1.0 / (ratio + 0.01) if ratio > 0 else 100  # Avoid division by zero
                    inverse_ratios[cls] = inverse
                    total_inverse_ratio += inverse
                
                # Allocate remaining budget proportionally to inverse ratios
                for cls in available_classes:
                    if total_inverse_ratio > 0:
                        additional = int(np.floor(remaining * inverse_ratios[cls] / total_inverse_ratio))
                        target_counts[cls] = target_counts.get(cls, 0) + additional
        
        # Second approach: Balance based on representation ratios
        else:
            # Calculate target based on future global distribution
            for cls in available_classes:
                # Target count for this class in the future labeled set
                target_global_count = self.global_class_distribution[cls] * future_size
                
                # Current count for this class
                current_count = current_distribution.get(cls, 0) * labeled_set_size
                
                # How many samples we need to add to reach target
                samples_needed = max(0, int(np.ceil(target_global_count - current_count)))
                target_counts[cls] = samples_needed
        
        # If the sum of target counts exceeds budget, adjust proportionally but prioritize underrepresented
        total_target = sum(target_counts.values())
        if total_target > num_samples:
            # Sort classes by representation ratio (lowest first = most underrepresented)
            sorted_classes = sorted(available_classes, key=lambda cls: representation_ratios.get(cls, float('inf')))
            
            # Reset target counts
            target_counts = {cls: 0 for cls in available_classes}
            
            # Distribute budget prioritizing most underrepresented classes
            remaining = num_samples
            for cls in sorted_classes:
                if remaining <= 0:
                    break
                
                # Allocate more aggressively to underrepresented classes
                ratio = representation_ratios.get(cls, 1.0)
                
                # Very underrepresented classes get more budget
                if ratio < 0.5:
                    # The lower the ratio, the higher percentage of remaining budget
                    percentage = 0.5  # At least 50% of remaining for very underrepresented
                elif ratio < 0.8:
                    percentage = 0.3  # 30% for moderately underrepresented
                else:
                    percentage = 0.2  # 20% for closer to balanced
                
                allocation = min(remaining, max(1, int(np.ceil(remaining * percentage))))
                target_counts[cls] = allocation
                remaining -= allocation
                
                if remaining <= 3 and remaining > 0:
                    for c in sorted_classes:
                        if remaining <= 0:
                            break
                        target_counts[c] = target_counts.get(c, 0) + 1
                        remaining -= 1
        
        # If there's still remaining budget, distribute it to maximize balance
        remaining = num_samples - sum(target_counts.values())
        if remaining > 0:
            
            # Sort by current representation after initial allocation
            # We need to recalculate the projected ratios after our initial allocation
            projected_counts = {}
            projected_ratios = {}
            
            for cls in available_classes:
                # Calculate the projected count after initial allocation
                current_count = current_distribution.get(cls, 0) * labeled_set_size
                projected_counts[cls] = current_count + target_counts.get(cls, 0)
                
                # Get the projected ratio compared to target
                projected_ratio = projected_counts[cls] / (future_size * self.global_class_distribution[cls]) \
                    if (future_size * self.global_class_distribution[cls]) > 0 else float('inf')
                projected_ratios[cls] = projected_ratio
            
            # Sort classes by projected ratio (lowest first)
            sorted_classes = sorted(available_classes, key=lambda cls: projected_ratios.get(cls, float('inf')))
            
            # Distribute remaining budget to most underrepresented classes first
            idx = 0
            while remaining > 0 and idx < len(sorted_classes):
                cls = sorted_classes[idx]
                target_counts[cls] = target_counts.get(cls, 0) + 1
                remaining -= 1
                
                # Cycle through underrepresented classes to maintain balance
                idx = (idx + 1) % len(sorted_classes)
                
                # If we've gone through all classes once, re-sort based on updated projections
                if idx == 0 and remaining > 0:
                    # Update projected counts and ratios
                    for c in available_classes:
                        current_count = current_distribution.get(c, 0) * labeled_set_size
                        projected_counts[c] = current_count + target_counts.get(c, 0)
                        projected_ratio = projected_counts[c] / (future_size * self.global_class_distribution[c]) \
                            if (future_size * self.global_class_distribution[c]) > 0 else float('inf')
                        projected_ratios[c] = projected_ratio
                    
                    # Re-sort based on updated projections
                    sorted_classes = sorted(available_classes, key=lambda c: projected_ratios.get(c, float('inf')))
        
        # Final verification that we're using exactly the budget
        final_count = sum(target_counts.values())
        if final_count != num_samples:
            
            # Force exact budget usage
            diff = num_samples - final_count
            if diff > 0:
                # Need to add more samples
                sorted_classes = sorted(available_classes, 
                                         key=lambda cls: representation_ratios.get(cls, float('inf')))
                for cls in sorted_classes:
                    if diff <= 0:
                        break
                    target_counts[cls] = target_counts.get(cls, 0) + 1
                    diff -= 1
            else:
                # Need to remove some samples
                sorted_classes = sorted(available_classes, 
                                         key=lambda cls: -representation_ratios.get(cls, float('inf')))
                for cls in sorted_classes:
                    if diff >= 0 or target_counts.get(cls, 0) <= 0:
                        break
                    target_counts[cls] = target_counts.get(cls, 0) - 1
                    diff += 1
        
        
        return target_counts

    def pseudo_label_data(self, model, model_server, unlabeled_loader, class_std_devs=None):
        """
        Assign pseudo-labels to the unlabeled data using the model predictions.
        For classes with std_dev < 15, use both local and server models for entropy.
        
        Args:
            model (torch.nn.Module): The client model to use for predictions.
            model_server (torch.nn.Module): The server model for low-variance classes.
            unlabeled_loader (DataLoader): Loader for unlabeled data.
            class_std_devs (dict, optional): Standard deviations by class.
            
        Returns:
            tuple: (indices, pseudo_labels, confidence_scores, entropy_scores)
        """
        
        # Track which classes have low std_dev (< 12)
        low_variance_classes = set()
        if class_std_devs is not None:
            for cls, std_dev in class_std_devs.items():
                if std_dev < 12.0:  
                    low_variance_classes.add(cls)
        
        # Collect predictions
        model.eval()
        model_server.eval()
        indices = []
        pseudo_labels = []
        confidence_scores = []
        entropy_scores = []
        
        batch_idx = 0
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                batch_idx += 1
                
                # Get batch indices
                batch_indices = unlabeled_loader.sampler.indices[
                    (batch_idx - 1) * unlabeled_loader.batch_size:
                    min(batch_idx * unlabeled_loader.batch_size, len(unlabeled_loader.sampler))
                ]
                indices.extend(batch_indices)
                
                # Forward pass with local model
                inputs = inputs.to(self.device)
                outputs_local = model(inputs)
                if isinstance(outputs_local, tuple):
                    outputs_local = outputs_local[0]
                
                # Get predicted classes and confidence scores from local model
                log_probs_local = F.log_softmax(outputs_local, dim=1)
                probs_local = torch.exp(log_probs_local)
                confidence_local, preds_local = torch.max(probs_local, dim=1)
                
                # Calculate entropy from local model
                batch_entropy_local = -torch.sum(probs_local * log_probs_local, dim=1)
                
                # For low variance classes, we also use the server model
                if low_variance_classes and model_server is not None:
                    # Forward pass with server model
                    outputs_server = model_server(inputs)
                    if isinstance(outputs_server, tuple):
                        outputs_server = outputs_server[0]
                    
                    # Get predicted classes and confidence scores from server model
                    log_probs_server = F.log_softmax(outputs_server, dim=1)
                    probs_server = torch.exp(log_probs_server)
                    
                    # Calculate entropy from server model
                    batch_entropy_server = -torch.sum(probs_server * log_probs_server, dim=1)
                    
                    # Combine entropies depending on class
                    batch_entropy = batch_entropy_local.clone()  # Start with local entropy
                    server_preds = torch.argmax(probs_server, dim=1)
                    
                    # For each sample, if its prediction is a low variance class, use average entropy
                    for i, (local_pred, server_pred) in enumerate(zip(preds_local, server_preds)):
                        if local_pred.item() in low_variance_classes or server_pred.item() in low_variance_classes:
                            # Use average of local and server entropy
                            batch_entropy[i] = (batch_entropy_local[i] + batch_entropy_server[i]) / 2.0
                else:
                    # Just use local model's entropy for all samples
                    batch_entropy = batch_entropy_local
                
                # Store results
                pseudo_labels.extend(preds_local.cpu().numpy())
                confidence_scores.extend(confidence_local.cpu().numpy())
                entropy_scores.extend(batch_entropy.cpu().numpy())
        
        # Convert to numpy arrays
        indices = np.array(indices)
        pseudo_labels = np.array(pseudo_labels)
        confidence_scores = np.array(confidence_scores)
        entropy_scores = np.array(entropy_scores)
        
        # Count pseudo-labels by class
        pseudo_counts = {}
        for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
            pseudo_counts[cls] = np.sum(pseudo_labels == cls)
        
        for cls in range(config.NUM_CLASSES):  
            count = pseudo_counts[cls]
            percentage = count / len(pseudo_labels) * 100 if len(pseudo_labels) > 0 else 0
            avg_entropy = np.mean(entropy_scores[pseudo_labels == cls]) if count > 0 else 0
            is_low_var = "Low variance - combined models" if cls in low_variance_classes else "Normal"
        
        return indices, pseudo_labels, confidence_scores, entropy_scores

    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, num_samples, labeled_set=None, seed=None, global_class_distribution=None, class_variance_stats=None):
        """
        Selects samples using pseudo-class-balanced entropy-based sampling.
        
        Args:
            model (torch.nn.Module): Client model.
            model_server (torch.nn.Module): Server model (not used for pseudo-labeling).
            unlabeled_loader (DataLoader): Loader for unlabeled data.
            client_id (int): ID of the client.
            unlabeled_set (list): List of indices of unlabeled samples.
            num_samples (int): Number of samples to select.
            labeled_set (list, optional): List of indices of labeled samples.
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
         # Comprehensive seed setting for reproducibility
        if seed is not None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Get dataset for accessing labels
        dataset = unlabeled_loader.dataset

        # Use the provided labeled_set which should always be passed
        if labeled_set is not None:
            print(f"[AHFAL] Using provided labeled set with {len(labeled_set)} samples")
        else:
            labeled_set = []
            print(f"[AHFAL] No labeled set provided, assuming empty labeled set")

        # Track current cycle for this client
        if not hasattr(self, 'client_cycles'):
            self.client_cycles = {}
        self.client_cycles[client_id] = self.client_cycles.get(client_id, 0) + 1
        
        # For tracking selections across cycles
        if not hasattr(self, 'client_labeled_sets'):
            self.client_labeled_sets = {}
        
        if client_id not in self.client_labeled_sets:
            self.client_labeled_sets[client_id] = []
        
        # Print class variance stats across clients if provided
        if class_variance_stats is not None and isinstance(class_variance_stats, dict) and 'class_stats' in class_variance_stats:
            class_stats = class_variance_stats['class_stats']
            
            # Print per-class standard deviation
            for cls in sorted(class_stats.keys()):
                stats_dict = class_stats[cls]
                std_dev = stats_dict.get('std_dev', 0)
                if hasattr(std_dev, 'item'):
                    std_dev = std_dev.item()
        
        # Calculate local class distribution using true labels
        local_distribution = self.calculate_local_class_distribution(labeled_set, dataset)
        
        # Use the global distribution provided by the trainer if available
        if global_class_distribution is not None:
            self.global_class_distribution = global_class_distribution
            
            # Print the global distribution
            print("[AHFAL] Global class distribution:")
            for cls in range(config.NUM_CLASSES):
                percentage = global_class_distribution.get(cls, 0) * 100
                print(f"  Class {cls}: {percentage:.2f}%")
                
        else:
            # If global distribution is not available, raise an error
            raise ValueError("[AHFAL] Error: Global class distribution not provided. Cannot proceed without global distribution.")
        
        # Step 2: Get labeled sample class distribution using pseudo-labels from the server model
        labeled_pseudo_counts = {i: 0 for i in range(config.NUM_CLASSES)}
        total_labeled = len(labeled_set)
        
        if total_labeled > 0:
            # Create loader for labeled set
            labeled_loader = DataLoader(
                dataset,
                batch_size=64,
                sampler=SubsetSequentialSampler(labeled_set),
                num_workers=2,
                pin_memory=True
            )
            
            # Get pseudo-labels for labeled set using the server model
            model_server.eval()
            with torch.no_grad():
                for inputs, _ in labeled_loader:
                    inputs = inputs.to(self.device)
                    outputs = model_server(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    _, preds = torch.max(outputs, dim=1)
                    for pred in preds.cpu().numpy():
                        labeled_pseudo_counts[pred] += 1
            
            # Print the labeled set pseudo-class distribution
            for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
                percentage = (labeled_pseudo_counts.get(cls, 0) / total_labeled * 100) if total_labeled > 0 else 0
                # print(f"  Class {cls}: {labeled_pseudo_counts.get(cls, 0)} samples ({percentage:.1f}%)")
        
        # Calculate the current class distribution percentages
        current_distribution = {cls: count/total_labeled for cls, count in labeled_pseudo_counts.items()} if total_labeled > 0 else {cls: 0 for cls in range(config.NUM_CLASSES)}
        
        # Get class std_devs from variance stats (if provided)
        class_std_devs = None
        if class_variance_stats is not None and isinstance(class_variance_stats, dict) and 'class_stats' in class_variance_stats:
            class_std_devs = {}
            for cls, stats in class_variance_stats['class_stats'].items():
                std_dev = stats.get('std_dev', 0)
                if hasattr(std_dev, 'item'):
                    std_dev = std_dev.item()
                std_dev = std_dev * 100  # Convert to percentage
                class_std_devs[cls] = std_dev
        
        # Step 3: Generate pseudo-labels for unlabeled set
        indices, pseudo_labels, confidence_scores, entropy_scores = self.pseudo_label_data(
            model=model,  # Use local model for pseudo-labeling
            model_server=model_server,  # Pass server model for low-variance classes
            unlabeled_loader=unlabeled_loader,
            class_std_devs=class_std_devs  # Pass class standard deviations
        )
        
        # Step 4: Get all available pseudo-classes in the unlabeled pool
        available_classes = set(pseudo_labels)
        
        # Step 5: Calculate the target counts for each class
        target_counts = self.compute_target_counts(
            current_distribution, 
            num_samples, 
            total_labeled, 
            available_classes
        )
        
        # Step 6: Organize samples by pseudo-class with their entropy scores
        class_entropy_mapping = {}
        for i, (idx, label, entropy) in enumerate(zip(indices, pseudo_labels, entropy_scores)):
            if label not in class_entropy_mapping:
                class_entropy_mapping[label] = []
            class_entropy_mapping[label].append((idx, entropy))
        
        # Count available samples by pseudo-class
        available_by_class = {cls: len(samples) for cls, samples in class_entropy_mapping.items() if len(samples) > 0}
        
        # Step 7: Check if target counts are achievable
        for cls, count in target_counts.items():
            available = available_by_class.get(cls, 0)
            if count > available:
                # Adjust target count to available
                target_counts[cls] = available
        
        # Step 8: After adjustment, redistribute any unused budget
        total_adjusted = sum(target_counts.values())
        if total_adjusted < num_samples:
            # Find classes that can take more samples
            extra_capacity = {}
            for cls in available_classes:
                extra = available_by_class.get(cls, 0) - target_counts.get(cls, 0)
                if extra > 0:
                    extra_capacity[cls] = extra
            
            # Redistribute to classes with extra capacity, prioritizing most underrepresented
            if extra_capacity:
                remaining = num_samples - total_adjusted
                # Sort classes by representation ratio
                if total_labeled > 0:
                    sorted_classes = sorted(
                        extra_capacity.keys(),
                        key=lambda c: current_distribution.get(c, 0) / self.global_class_distribution[c] \
                            if self.global_class_distribution[c] > 0 else float('inf')
                    )
                else:
                    # If no labeled data yet, distribute evenly
                    sorted_classes = list(extra_capacity.keys())
                
                # Distribute remaining budget
                for cls in sorted_classes:
                    take = min(remaining, extra_capacity[cls])
                    target_counts[cls] = target_counts.get(cls, 0) + take
                    remaining -= take
                    if remaining <= 0:
                        break
        
        # Step 9: Select samples based on entropy within each pseudo-class
        selected_samples = []
        balanced_selections = {}
        
        # Process classes in order of target count (highest first) for logging clarity
        sorted_classes = sorted(target_counts.keys(), key=lambda cls: target_counts.get(cls, 0), reverse=True)
        
        for cls in sorted_classes:
            if target_counts[cls] > 0:
                if cls not in class_entropy_mapping or len(class_entropy_mapping[cls]) == 0:
                    continue
                
                # Sort samples by entropy (highest first)
                class_samples = class_entropy_mapping[cls]
                class_samples.sort(key=lambda x: x[1], reverse=True)  # Highest entropy first
                
                # Select top samples by entropy
                num_to_select = min(target_counts[cls], len(class_samples))
                selected_indices = [sample[0] for sample in class_samples[:num_to_select]]
                selected_samples.extend(selected_indices)
                
                balanced_selections[cls] = num_to_select
        
        # Step 10: Check if we need to handle unallocated budget
        remaining_to_select = num_samples - len(selected_samples)
        if remaining_to_select > 0:
            
            # As a last resort, just select any remaining samples by entropy
            remaining_indices = [idx for idx in unlabeled_set if idx not in selected_samples]
            if remaining_indices:
                # Map back to original indices
                remaining_mapping = {}
                for i, idx in enumerate(indices):
                    if idx in remaining_indices:
                        remaining_mapping[idx] = entropy_scores[i]
                
                # Sort by entropy (highest first)
                sorted_remaining = sorted(remaining_mapping.items(), key=lambda x: x[1], reverse=True)
                additional = min(remaining_to_select, len(sorted_remaining))
                additional_indices = [idx for idx, _ in sorted_remaining[:additional]]
                selected_samples.extend(additional_indices)
        

        # Calculate remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        # Track selections
        self.client_labeled_sets[client_id].extend(selected_samples)
        
        # Debug info about final selection
        # Map selected samples to their pseudo-labels
        selected_pseudo_classes = []
        for idx in selected_samples:
            found = False
            for i, original_idx in enumerate(indices):
                if original_idx == idx:
                    selected_pseudo_classes.append(pseudo_labels[i])
                    found = True
                    break
            if not found:
                selected_pseudo_classes.append(-1)  # Unknown class if not found
        
        final_class_counts = {}
        for cls in range(config.NUM_CLASSES):  
            final_class_counts[cls] = sum(1 for label in selected_pseudo_classes if label == cls)
        
        for cls in range(config.NUM_CLASSES):  
            count = final_class_counts.get(cls, 0)
            percentage = count / len(selected_samples) * 100 if len(selected_samples) > 0 else 0
        
        
        # Calculate the new distribution after this selection
        future_distribution = {}
        future_size = total_labeled + len(selected_samples)
        for cls in range(config.NUM_CLASSES):  
            # Add newly selected samples to current distribution
            future_distribution[cls] = (labeled_pseudo_counts.get(cls, 0) + final_class_counts.get(cls, 0)) / future_size
        
        # Calculate how close we got to the global distribution
        dist_error = sum(abs(future_distribution.get(cls, 0) - self.global_class_distribution.get(cls, 0)) for cls in range(config.NUM_CLASSES)) / 2
        
        # Display improvement compared to initial distribution
        if total_labeled > 0:
            initial_distribution = {cls: count/total_labeled for cls, count in labeled_pseudo_counts.items()}
            initial_error = sum(abs(initial_distribution.get(cls, 0) - self.global_class_distribution.get(cls, 0)) for cls in range(config.NUM_CLASSES)) / 2
            improvement = initial_error - dist_error
        
        return selected_samples, remaining_unlabeled