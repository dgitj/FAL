import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
import config  # Import config to access NUM_CLASSES

class PseudoClassBalancedConfidenceSampler:
    def __init__(self, device="cuda"):
        """
        Initializes the Pseudo-Class-Balanced Confidence sampler.
        This sampler uses the global model for pseudo-labeling data and balancing class distribution,
        selecting samples with the highest confidence per pseudo-class.

        Args:
            device (str): Device to run calculations on (e.g., 'cuda' or 'cpu').
        """
        # Check if CUDA is available, fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            print("[PseudoConfidence] CUDA not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
            
        self.debug = True  # Enable detailed debugging
        
        # We'll estimate the global class distribution from data in each cycle
        # so we don't need to initialize it here
        self.global_class_distribution = None
        print(f"[PseudoConfidence] Using device: {self.device}")
        
        # For tracking client progress across cycles
        self.client_cycles = {}
        self.client_labeled_sets = {}

    def estimate_global_distribution(self, model, dataset, sample_size=10000, seed=42):
        """
        Estimate the global class distribution using the global model for pseudo-labeling.
        
        Args:
            model (torch.nn.Module): The global model to use for pseudo-labeling.
            dataset: The full dataset.
            sample_size (int): Number of samples to use for estimating distribution.
            seed (int): Random seed for reproducibility.
            
        Returns:
            dict: Estimated global class distribution.
        """
        print(f"[PseudoConfidence] Estimating global class distribution using {sample_size} samples")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Sample random indices from the dataset
        all_indices = list(range(len(dataset)))
        sampled_indices = np.random.choice(all_indices, min(sample_size, len(dataset)), replace=False)
        
        # Create a loader for the sampled data
        sampler = SubsetSequentialSampler(sampled_indices)
        loader = DataLoader(
            dataset,
            batch_size=64,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Collect predictions
        model.eval()
        all_preds = []
        all_confidence = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                
                # Get model predictions
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Get predicted classes and confidence scores
                probs = F.softmax(outputs, dim=1)
                confidence, preds = torch.max(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_confidence.extend(confidence.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_confidence = np.array(all_confidence)
        
        # Count occurrences of each class
        class_counts = {}
        for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
            class_counts[cls] = np.sum(all_preds == cls)
        
        # Calculate distribution
        total_samples = len(all_preds)
        distribution = {cls: count / total_samples for cls, count in class_counts.items()}
        
        print("[PseudoConfidence] Estimated global class distribution:")
        for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
            print(f"  Class {cls}: {distribution[cls]:.4f} ({class_counts[cls]} samples)")
        
        return distribution

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
        
        # Safety check - if global_class_distribution is not set yet, use uniform distribution
        if self.global_class_distribution is None:
            self.global_class_distribution = {i: 1.0/config.NUM_CLASSES for i in range(config.NUM_CLASSES)}
            print(f"[PseudoConfidence] Using uniform distribution as fallback")
        
        if self.debug:
            print(f"[PseudoConfidence] Planning to select {num_samples} samples")
            print(f"[PseudoConfidence] Future labeled set size will be: {future_size}")
            print(f"[PseudoConfidence] Available classes in unlabeled pool: {sorted(list(available_classes))}")  
        
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
                    percentage = 0.5  # At least 50% of remaining for very underrepresented
                elif ratio < 0.8:
                    percentage = 0.3  # 30% for moderately underrepresented
                else:
                    percentage = 0.2  # 20% for closer to balanced
                
                allocation = min(remaining, max(1, int(np.ceil(remaining * percentage))))
                target_counts[cls] = allocation
                remaining -= allocation
                
                # If we're down to the last few samples, just allocate them one by one
                if remaining <= 3 and remaining > 0:
                    for c in sorted_classes:
                        if remaining <= 0:
                            break
                        target_counts[c] = target_counts.get(c, 0) + 1
                        remaining -= 1
        
        # If there's still remaining budget, distribute it to maximize balance
        remaining = num_samples - sum(target_counts.values())
        if remaining > 0:
            if self.debug:
                print(f"[PseudoConfidence] Distributing {remaining} remaining samples to maximize balance")
            
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
            print(f"[PseudoConfidence] WARNING: Target count {final_count} doesn't match budget {num_samples}")
            
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
            print(f"[PseudoConfidence] Target counts per class: {target_counts}")
            print(f"[PseudoConfidence] Total samples to select: {sum(target_counts.values())}")
        
        return target_counts

    def pseudo_label_data(self, model, unlabeled_loader):
        """
        Assign pseudo-labels to the unlabeled data using the model predictions.
        
        Args:
            model (torch.nn.Module): The model to use for predictions.
            unlabeled_loader (DataLoader): Loader for unlabeled data.
            
        Returns:
            tuple: (indices, pseudo_labels, confidence_scores, entropy_scores)
        """
        print(f"[PseudoConfidence] Assigning pseudo-labels to unlabeled data")
        
        # Collect predictions
        model.eval()
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
                
                # Forward pass
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Get predicted classes and confidence scores
                log_probs = F.log_softmax(outputs, dim=1)
                probs = torch.exp(log_probs)
                confidence, preds = torch.max(probs, dim=1)
                
                # Calculate entropy
                batch_entropy = -torch.sum(probs * log_probs, dim=1)
                
                # Store results
                pseudo_labels.extend(preds.cpu().numpy())
                confidence_scores.extend(confidence.cpu().numpy())
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
        
        print(f"[PseudoConfidence] Pseudo-label distribution:")
        for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
            count = pseudo_counts[cls]
            percentage = count / len(pseudo_labels) * 100 if len(pseudo_labels) > 0 else 0
            avg_confidence = np.mean(confidence_scores[pseudo_labels == cls]) if count > 0 else 0
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%), avg confidence: {avg_confidence:.3f}")
        
        return indices, pseudo_labels, confidence_scores, entropy_scores

    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, num_samples, labeled_set=None, seed=None):
        """
        Selects samples using pseudo-class-balanced confidence-based sampling.
        
        Args:
            model (torch.nn.Module): Client model.
            model_server (torch.nn.Module): Server model to use for pseudo-labeling.
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
            print(f"\n[PseudoConfidence] Client {client_id}: Selecting {num_samples} samples")
            print(f"[PseudoConfidence] Unlabeled pool size: {len(unlabeled_set)}")
        
        # Get dataset for accessing labels (only for statistics)
        dataset = unlabeled_loader.dataset

        # Use the provided labeled_set which should always be passed
        if labeled_set is not None:
            print(f"[PseudoConfidence] Using provided labeled set with {len(labeled_set)} samples")
        else:
            labeled_set = []
            print(f"[PseudoConfidence] No labeled set provided, assuming empty labeled set")

        # Track current cycle for this client
        if not hasattr(self, 'client_cycles'):
            self.client_cycles = {}
        self.client_cycles[client_id] = self.client_cycles.get(client_id, 0) + 1
        
        # For tracking selections across cycles
        if not hasattr(self, 'client_labeled_sets'):
            self.client_labeled_sets = {}
        
        if client_id not in self.client_labeled_sets:
            self.client_labeled_sets[client_id] = []
        
        # Step 1: Estimate global class distribution using the server model every cycle
        print(f"[PseudoConfidence] Cycle {self.client_cycles.get(client_id, 0)} - Estimating global distribution")
        self.global_class_distribution = self.estimate_global_distribution(
            model=model_server,  # Use server model
            dataset=dataset,
            sample_size=10000,  # Use more samples for a better estimate
            seed=42 + (client_id if seed is None else seed) + self.client_cycles.get(client_id, 0)*100  # Different seed per client and cycle
        )
        
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
            print(f"[PseudoConfidence] Labeled set pseudo-class distribution:")
            for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
                percentage = (labeled_pseudo_counts.get(cls, 0) / total_labeled * 100) if total_labeled > 0 else 0
                print(f"  Class {cls}: {labeled_pseudo_counts.get(cls, 0)} samples ({percentage:.1f}%)")
        
        # Calculate the current class distribution percentages
        current_distribution = {cls: count/total_labeled for cls, count in labeled_pseudo_counts.items()} if total_labeled > 0 else {cls: 0 for cls in range(config.NUM_CLASSES)}
        
        # Step 3: Generate pseudo-labels for unlabeled set
        indices, pseudo_labels, confidence_scores, entropy_scores = self.pseudo_label_data(
            model=model_server,  # Use server model for consistent pseudo-labeling
            unlabeled_loader=unlabeled_loader
        )
        
        # Step 4: No filtering by confidence threshold - using all samples
        print(f"[PseudoConfidence] Using all {len(indices)} samples for selection")
        
        # Get all available pseudo-classes in the unlabeled pool
        available_classes = set(pseudo_labels)
        print(f"[PseudoConfidence] Available pseudo-classes in unlabeled pool: {sorted(list(available_classes))}")
        
        # Step 5: Calculate the target counts for each class
        target_counts = self.compute_target_counts(
            current_distribution, 
            num_samples, 
            total_labeled, 
            available_classes
        )
        
        # Step 6: Organize samples by pseudo-class with their confidence scores
        class_confidence_mapping = {}
        for i, (idx, label, confidence) in enumerate(zip(indices, pseudo_labels, confidence_scores)):
            if label not in class_confidence_mapping:
                class_confidence_mapping[label] = []
            class_confidence_mapping[label].append((idx, confidence))
        
        # Count available samples by pseudo-class
        available_by_class = {cls: len(samples) for cls, samples in class_confidence_mapping.items() if len(samples) > 0}
        print(f"[PseudoConfidence] Available samples by pseudo-class: {available_by_class}")
        
        # Step 7: Check if target counts are achievable
        for cls, count in target_counts.items():
            available = available_by_class.get(cls, 0)
            if count > available:
                print(f"[PseudoConfidence] Warning: Target count {count} for pseudo-class {cls} exceeds available {available}")
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
                
                print(f"[PseudoConfidence] Redistributed {num_samples - total_adjusted} samples, new target counts: {target_counts}")
        
        # Step 9: Select samples based on confidence within each pseudo-class (KEY DIFFERENCE FROM ENTROPY)
        selected_samples = []
        balanced_selections = {}
        
        # Process classes in order of target count (highest first) for logging clarity
        sorted_classes = sorted(target_counts.keys(), key=lambda cls: target_counts.get(cls, 0), reverse=True)
        
        for cls in sorted_classes:
            if target_counts[cls] > 0:
                if cls not in class_confidence_mapping or len(class_confidence_mapping[cls]) == 0:
                    print(f"[PseudoConfidence] No unlabeled samples available for pseudo-class {cls}")
                    continue
                
                # Sort samples by confidence (highest first) - KEY DIFFERENCE FROM ENTROPY
                class_samples = class_confidence_mapping[cls]
                class_samples.sort(key=lambda x: x[1], reverse=True)  # Highest confidence first
                
                # Select top samples by confidence
                num_to_select = min(target_counts[cls], len(class_samples))
                selected_indices = [sample[0] for sample in class_samples[:num_to_select]]
                selected_samples.extend(selected_indices)
                
                balanced_selections[cls] = num_to_select
                print(f"[PseudoConfidence] Selected {num_to_select} samples from pseudo-class {cls}")
        
        # Step 10: Check if we need to handle unallocated budget
        remaining_to_select = num_samples - len(selected_samples)
        if remaining_to_select > 0:
            print(f"[PseudoConfidence] WARNING: Still need to select {remaining_to_select} samples")
            
            # As a last resort, just select any remaining samples by confidence
            remaining_indices = [idx for idx in unlabeled_set if idx not in selected_samples]
            if remaining_indices:
                # Map back to original indices
                remaining_mapping = {}
                for i, idx in enumerate(indices):
                    if idx in remaining_indices:
                        remaining_mapping[idx] = confidence_scores[i]
                
                # Sort by confidence (highest first)
                sorted_remaining = sorted(remaining_mapping.items(), key=lambda x: x[1], reverse=True)
                additional = min(remaining_to_select, len(sorted_remaining))
                additional_indices = [idx for idx, _ in sorted_remaining[:additional]]
                selected_samples.extend(additional_indices)
                
                print(f"[PseudoConfidence] Selected {additional} last-resort samples based on confidence")
        
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
        for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
            final_class_counts[cls] = sum(1 for label in selected_pseudo_classes if label == cls)
        
        print(f"\n[PseudoConfidence] Final selection pseudo-class distribution:")
        for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
            count = final_class_counts.get(cls, 0)
            percentage = count / len(selected_samples) * 100 if len(selected_samples) > 0 else 0
            target = target_counts.get(cls, 0)
            print(f"  Class {cls}: {count} samples ({percentage:.1f}%) [Target: {target}]")
        
        print(f"[PseudoConfidence] Total selected: {len(selected_samples)} out of budget {num_samples}")
        
        # Calculate the new distribution after this selection
        future_distribution = {}
        future_size = total_labeled + len(selected_samples)
        for cls in range(config.NUM_CLASSES):  # Use NUM_CLASSES from config
            # Add newly selected samples to current distribution
            future_distribution[cls] = (labeled_pseudo_counts.get(cls, 0) + final_class_counts.get(cls, 0)) / future_size
        
        # Calculate how close we got to the global distribution
        dist_error = sum(abs(future_distribution.get(cls, 0) - self.global_class_distribution.get(cls, 0)) for cls in range(config.NUM_CLASSES)) / 2
        print(f"[PseudoConfidence] Distribution error after selection: {dist_error:.4f} (lower is better)")
        
        # Display improvement compared to initial distribution
        if total_labeled > 0:
            initial_distribution = {cls: count/total_labeled for cls, count in labeled_pseudo_counts.items()}
            initial_error = sum(abs(initial_distribution.get(cls, 0) - self.global_class_distribution.get(cls, 0)) for cls in range(config.NUM_CLASSES)) / 2
            improvement = initial_error - dist_error
            print(f"[PseudoConfidence] Initial error: {initial_error:.4f}, Improvement: {improvement:.4f} ({improvement/initial_error*100:.1f}% better)")
        
        return selected_samples, remaining_unlabeled
