import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Subset

class ClassBalancedCoreSetSampler:
    def __init__(self, device="cuda"):
        """
        Initializes the Class-Balanced CoreSet sampler with privileged information.
        This sampler uses ground truth labels to balance class distribution and 
        CoreSet diversity selection within each class.

        Args:
            device (str): Device to run calculations on (e.g., 'cuda' or 'cpu').
        """
        # Check if CUDA is available, fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            print("[ClassBalancedCoreSet] CUDA not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
            
        self.debug = True  # Enable detailed debugging
        
        # Set the global class distribution for CIFAR-10 (uniform distribution)
        # CIFAR-10 has 10 classes with equal distribution
        self.global_class_distribution = {i: 0.1 for i in range(10)}
        
        print("[ClassBalancedCoreSet] Initialized with global distribution:", self.global_class_distribution)
        print(f"[ClassBalancedCoreSet] Using device: {self.device}")

    def _extract_features(self, model, loader):
        """
        Extract features for all samples in the loader using the model.
        
        Args:
            model: PyTorch model to extract features from.
            loader: DataLoader containing samples.
            
        Returns:
            numpy.ndarray: Features for all samples.
            list: Indices of the samples in the original dataset
        """
        model.eval()
        features = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                
                # Forward pass to get features
                # The model returns (scores, features) where features is a list
                outputs = model(inputs)
                
                # Extract features based on model output format
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    # Get the features (second element of tuple)
                    batch_features = outputs[1]
                    
                    # In this model implementation, the features are a list of tensors
                    # We'll use the last element (final layer features)
                    if isinstance(batch_features, list) and len(batch_features) > 0:
                        # Use the last element (index -1) which should be the final feature representation
                        batch_features = batch_features[-1]  # This is typically the flattened features before FC layer
                    
                else:
                    # Fallback if model doesn't return expected format
                    batch_features = outputs
                
                # Store results
                features.append(batch_features.cpu().numpy())
                
        # Concatenate results from all batches
        if len(features) > 0:
            return np.concatenate(features)
        else:
            return np.array([])
    
    def _greedy_k_center(self, labeled_features, unlabeled_features, unlabeled_indices, budget):
        """
        Greedy k-center algorithm to select samples.
        
        Args:
            labeled_features (np.ndarray): Features of labeled samples.
            unlabeled_features (np.ndarray): Features of unlabeled samples.
            unlabeled_indices (list): Indices of unlabeled samples.
            budget (int): Number of samples to select.
            
        Returns:
            list: Indices of selected samples.
        """
        if budget <= 0:
            return []
            
        if len(unlabeled_indices) <= budget:
            return unlabeled_indices
            
        selected = []
        
        # Calculate minimum distances for each unlabeled point to any labeled point
        if len(labeled_features) > 0:
            # If we have labeled samples, compute distances to them
            min_distances = pairwise_distances(unlabeled_features, labeled_features, metric='euclidean').min(axis=1)
        else:
            # If no labeled samples yet, pick the first point arbitrarily
            # and calculate distances to an arbitrarily chosen point (the first point)
            min_distances = pairwise_distances(unlabeled_features, unlabeled_features[0:1], metric='euclidean').min(axis=1)
        
        # Greedy selection
        for _ in range(budget):
            if len(min_distances) == 0:
                break
                
            # Find the point that has the maximum minimum distance
            idx = np.argmax(min_distances)
            
            # Add this point to the selected set
            selected.append(unlabeled_indices[idx])
            
            # Recalculate minimum distances with the new point included
            new_distances = pairwise_distances(unlabeled_features, unlabeled_features[idx:idx+1], metric='euclidean').reshape(-1)
            min_distances = np.minimum(min_distances, new_distances)
        
        return selected

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
        
        if self.debug:
            print(f"[ClassBalancedCoreSet] Planning to select {num_samples} samples")
            print(f"[ClassBalancedCoreSet] Future labeled set size will be: {future_size}")
            print(f"[ClassBalancedCoreSet] Available classes in unlabeled pool: {sorted(list(available_classes))}")  
        
        # Check if all available classes have the same representation
        available_class_counts = {}
        for cls in available_classes:
            available_class_counts[cls] = current_distribution.get(cls, 0) * labeled_set_size
        
        # Calculate representation ratios for available classes
        # This measures how well each class is represented compared to its target
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
                    percentage = 0.9  # 90% of remaining for very underrepresented
                elif ratio < 0.8:
                    percentage = 0.1  # 10% for moderately underrepresented
                else:
                    percentage = 0.0  # 0% for closer to balanced
                
                allocation = min(remaining, int(np.ceil(remaining * percentage)))
                target_counts[cls] = allocation
                remaining -= allocation
        
        # If there's still remaining budget, distribute it to maximize balance
        remaining = num_samples - sum(target_counts.values())
        if remaining > 0:
            if self.debug:
                print(f"[ClassBalancedCoreSet] Distributing {remaining} remaining samples to maximize balance")
            
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
            print(f"[ClassBalancedCoreSet] WARNING: Target count {final_count} doesn't match budget {num_samples}")
            
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
        
        if self.debug:
            print(f"[ClassBalancedCoreSet] Target counts per class: {target_counts}")
            print(f"[ClassBalancedCoreSet] Total samples to select: {sum(target_counts.values())}")
        
        total_target = sum(target_counts.values())
        print(f"[ClassBalancedCoreSet] Total allocation for balancing: {total_target} (expected {num_samples})")
        
        return target_counts

    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, num_samples, labeled_set=None, seed=None):
        """
        Selects samples using class-balanced CoreSet-based sampling.
        
        Args:
            model (torch.nn.Module): Client model.
            model_server (torch.nn.Module): Server model (not used in this strategy).
            unlabeled_loader (DataLoader): Loader for unlabeled data.
            client_id (int): ID of the client.
            unlabeled_set (list): List of indices of unlabeled samples.
            num_samples (int): Number of samples to select.
            labeled_set (list, optional): List of indices of labeled samples.
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        if self.debug:
            print(f"\n[ClassBalancedCoreSet] Client {client_id}: Selecting {num_samples} samples")
            print(f"[ClassBalancedCoreSet] Unlabeled pool size: {len(unlabeled_set)}")
        
        # Get dataset for accessing labels
        dataset = unlabeled_loader.dataset

        # Use the provided labeled_set which should always be passed
        if labeled_set is not None:
            print(f"[ClassBalancedCoreSet] Using provided labeled set with {len(labeled_set)} samples")
        else:
            raise ValueError("[ClassBalancedCoreSet] Error: No labeled set provided. Make sure to set labeled_set_list in strategy_manager.")

        # Initialize tracking variables
        labeled_counts = {}
        total_labeled = len(labeled_set)
        
        # Get labeled sample classes
        for cls in range(10):
            labeled_counts[cls] = sum(1 for idx in labeled_set if dataset.targets[idx] == cls)
        
        # Track current cycle for this client
        if not hasattr(self, 'client_cycles'):
            self.client_cycles = {}
        self.client_cycles[client_id] = self.client_cycles.get(client_id, 0) + 1
        
        # For tracking selections across cycles
        if not hasattr(self, 'client_labeled_sets'):
            self.client_labeled_sets = {}
        
        if client_id not in self.client_labeled_sets:
            self.client_labeled_sets[client_id] = []
        
        # Get ground truth labels for the unlabeled set
        unlabeled_labels = [dataset.targets[idx] for idx in unlabeled_set]
        
        # Get all available classes in the unlabeled pool
        available_classes = set(unlabeled_labels)
        
        # Print available classes
        print(f"[ClassBalancedCoreSet] Available classes in unlabeled pool: {sorted(list(available_classes))}")
            
        # Print the class distribution of the first 10% labeled samples (BASE samples)
        if self.client_cycles.get(client_id, 0) == 1:  # First cycle
            print(f"\n[ClassBalancedCoreSet] === CLIENT {client_id} INITIAL LABELED DATA ANALYSIS ====")
            print(f"[ClassBalancedCoreSet] Initial labeled set size: {total_labeled} samples")
            print(f"[ClassBalancedCoreSet] Class distribution:")
            for cls in range(10):
                percentage = (labeled_counts.get(cls, 0) / total_labeled * 100) if total_labeled > 0 else 0
                print(f"    Class {cls}: {labeled_counts.get(cls, 0)} samples ({percentage:.1f}%)")
                
            # Calculate how far the initial distribution is from global
            if total_labeled > 0:
                current_distribution = {cls: count/total_labeled for cls, count in labeled_counts.items()}
                dist_error = sum(abs(current_distribution.get(cls, 0) - self.global_class_distribution[cls]) for cls in range(10)) / 2
                print(f"[ClassBalancedCoreSet] Distribution error: {dist_error:.4f} (lower is better)")
                
                # Calculate number of classes missing
                missing_classes = sum(1 for cls in range(10) if labeled_counts.get(cls, 0) == 0)
                print(f"[ClassBalancedCoreSet] Missing classes: {missing_classes}/10")
            print("[ClassBalancedCoreSet] ================================================\n")
        
        # Calculate the current class distribution percentages
        current_distribution = {cls: count/total_labeled for cls, count in labeled_counts.items()} if total_labeled > 0 else {cls: 0 for cls in range(10)}
        
        # Print current distribution of labeled data
        print(f"[ClassBalancedCoreSet] Current class distribution:")
        for cls in range(10):
            percentage = current_distribution.get(cls, 0) * 100
            available = "(available)" if cls in available_classes else "(not available)"
            print(f"    Class {cls}: {labeled_counts.get(cls, 0)} samples ({percentage:.1f}%) {available}")
        
        # Calculate the target counts using our improved method
        target_counts = self.compute_target_counts(current_distribution, num_samples, total_labeled, available_classes)
        
        print(f"[ClassBalancedCoreSet] Target counts to add by class: {target_counts}")
        
        # Extract features for labeled data
        if labeled_set and len(labeled_set) > 0:
            # Create a DataLoader for the labeled data
            labeled_subset = Subset(dataset, labeled_set)
            labeled_loader = DataLoader(
                labeled_subset, 
                batch_size=unlabeled_loader.batch_size, 
                shuffle=False,
                num_workers=unlabeled_loader.num_workers if hasattr(unlabeled_loader, 'num_workers') else 0
            )
            
            # Extract features for labeled data
            labeled_features = self._extract_features(model, labeled_loader)
            print(f"[ClassBalancedCoreSet] Extracted features for {len(labeled_set)} labeled samples")
            
            # Group labeled features by class
            labeled_features_by_class = {}
            for idx, class_id in enumerate(dataset.targets[i] for i in labeled_set):
                if class_id not in labeled_features_by_class:
                    labeled_features_by_class[class_id] = []
                # Make sure the index is valid
                if idx < len(labeled_features):
                    labeled_features_by_class[class_id].append(labeled_features[idx])
            
            # Convert lists to numpy arrays
            for cls in labeled_features_by_class:
                if labeled_features_by_class[cls]:
                    labeled_features_by_class[cls] = np.vstack(labeled_features_by_class[cls])
                else:
                    labeled_features_by_class[cls] = np.array([])
        else:
            labeled_features_by_class = {cls: np.array([]) for cls in range(10)}
            print("[ClassBalancedCoreSet] No labeled data to extract features from")
        
        # Organize unlabeled samples by class with their features
        # Extract features for all unlabeled samples
        unlabeled_features = self._extract_features(model, unlabeled_loader)
        
        # Group unlabeled samples by class
        class_mapping = {}
        for i, (idx, label) in enumerate(zip(unlabeled_set, unlabeled_labels)):
            if label not in class_mapping:
                class_mapping[label] = {'indices': [], 'features': []}
            class_mapping[label]['indices'].append(idx)
            if i < len(unlabeled_features):  # Safety check
                class_mapping[label]['features'].append(unlabeled_features[i])
        
        # Convert lists to numpy arrays
        for cls in class_mapping:
            if class_mapping[cls]['features']:
                class_mapping[cls]['features'] = np.vstack(class_mapping[cls]['features'])
        
        # Count available samples by class
        available_by_class = {cls: len(info['indices']) for cls, info in class_mapping.items()}
        print(f"[ClassBalancedCoreSet] Available unlabeled samples by class: {available_by_class}")
        
        # Check if target counts are achievable
        for cls, count in target_counts.items():
            available = available_by_class.get(cls, 0)
            if count > available:
                print(f"[ClassBalancedCoreSet] Warning: Target count {count} for class {cls} exceeds available {available}")
                # Adjust target count to available
                target_counts[cls] = available
        
        # After adjustment, redistribute any unused budget
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
        
        # Select samples using CoreSet within each class
        selected_samples = []
        balanced_selections = {}
        
        # Process classes in order of target count (highest first) for logging clarity
        sorted_classes = sorted(target_counts.keys(), key=lambda cls: target_counts.get(cls, 0), reverse=True)
        
        for cls in sorted_classes:
            if target_counts[cls] > 0:
                if cls not in class_mapping or len(class_mapping[cls]['indices']) == 0:
                    print(f"[ClassBalancedCoreSet] No unlabeled samples available for class {cls}")
                    continue
                
                # Get labeled features for this class
                labeled_class_features = labeled_features_by_class.get(cls, np.array([]))
                
                # Get unlabeled features and indices for this class
                unlabeled_class_features = class_mapping[cls]['features']
                unlabeled_class_indices = class_mapping[cls]['indices']
                
                # Select samples using CoreSet greedy k-center
                num_to_select = min(target_counts[cls], len(unlabeled_class_indices))
                
                if num_to_select > 0:
                    selected_class_indices = self._greedy_k_center(
                        labeled_class_features, 
                        unlabeled_class_features, 
                        unlabeled_class_indices, 
                        num_to_select
                    )
                    
                    selected_samples.extend(selected_class_indices)
                    balanced_selections[cls] = len(selected_class_indices)
                    print(f"[ClassBalancedCoreSet] Selected {len(selected_class_indices)} samples from class {cls}")
        
        # Calculate how much of the budget went to balanced selection
        balanced_count = len(selected_samples)
        balanced_percentage = balanced_count / num_samples * 100 if num_samples > 0 else 0
        print(f"[ClassBalancedCoreSet] Selected {balanced_count}/{num_samples} samples ({balanced_percentage:.1f}%) using class balancing")
        print(f"[ClassBalancedCoreSet] Class-balanced selections: {balanced_selections}")
        
        # Double-check: Are we missing any selection?
        remaining_to_select = num_samples - len(selected_samples)
        if remaining_to_select > 0:
            print(f"[ClassBalancedCoreSet] WARNING: Still need to select {remaining_to_select} samples")
            print(f"[ClassBalancedCoreSet] This should not happen with proper budget allocation!")
            
            # As a failsafe, select additional samples using CoreSet across all classes
            remaining_indices = [idx for idx in unlabeled_set if idx not in selected_samples]
            remaining_features = np.array([unlabeled_features[i] for i, idx in enumerate(unlabeled_set) if idx in remaining_indices])
            
            if len(remaining_indices) > 0 and len(remaining_features) > 0:
                # Select additional samples using CoreSet
                additional_samples = self._greedy_k_center(
                    labeled_features, 
                    remaining_features, 
                    remaining_indices, 
                    remaining_to_select
                )
                
                selected_samples.extend(additional_samples)
                
                additional_labels = [dataset.targets[idx] for idx in additional_samples]
                additional_by_class = {}
                for cls in range(10):
                    additional_by_class[cls] = sum(1 for label in additional_labels if label == cls)
                    
                print(f"[ClassBalancedCoreSet] FAILSAFE - Additional samples by class: {additional_by_class}")
                print(f"[ClassBalancedCoreSet] Selected {len(additional_samples)} additional samples using global CoreSet")
        
        # Calculate remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        # Debug info about final selection
        selected_classes = [dataset.targets[idx] for idx in selected_samples]
        final_class_counts = {}
        for cls in range(10):
            final_class_counts[cls] = sum(1 for label in selected_classes if label == cls)
        
        print(f"\n[ClassBalancedCoreSet] Final selection class distribution:")
        for cls in range(10):
            count = final_class_counts.get(cls, 0)
            percentage = count / len(selected_samples) * 100 if len(selected_samples) > 0 else 0
            target = target_counts.get(cls, 0)
            print(f"    Class {cls}: {count} samples ({percentage:.1f}%) [Target: {target}]")
            
        print(f"[ClassBalancedCoreSet] Total selected: {len(selected_samples)} out of budget {num_samples}")
        
        # Verify against target
        for cls, target in target_counts.items():
            actual = final_class_counts.get(cls, 0)
            if actual != target:
                print(f"[ClassBalancedCoreSet] Class {cls}: Selected {actual} vs target {target}")
        
        # Track selections for reference
        self.client_labeled_sets[client_id] = self.client_labeled_sets.get(client_id, []) + selected_samples
        
        # Calculate the new distribution after this selection
        future_distribution = {}
        future_size = total_labeled + len(selected_samples)
        for cls in range(10):
            future_distribution[cls] = (labeled_counts.get(cls, 0) + final_class_counts.get(cls, 0)) / future_size
        
        # Calculate how close we got to the global distribution
        dist_error = sum(abs(future_distribution.get(cls, 0) - self.global_class_distribution[cls]) for cls in range(10)) / 2
        print(f"[ClassBalancedCoreSet] Distribution error after selection: {dist_error:.4f} (lower is better)")
        
        # Display improvement compared to initial distribution
        if total_labeled > 0:
            initial_distribution = {cls: count/total_labeled for cls, count in labeled_counts.items()}
            initial_error = sum(abs(initial_distribution.get(cls, 0) - self.global_class_distribution[cls]) for cls in range(10)) / 2
            improvement = initial_error - dist_error
            print(f"[ClassBalancedCoreSet] Initial error: {initial_error:.4f}, Improvement: {improvement:.4f} ({improvement/initial_error*100:.1f}%)")
        
        return selected_samples, remaining_unlabeled