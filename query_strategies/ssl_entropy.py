import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.cluster import KMeans
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from models.ssl.contrastive_model import SimpleContrastiveLearning

# [ADDED] Import config to access global SSL settings
from config import USE_GLOBAL_SSL

class SSLEntropySampler:
    # [ADDED] Added global_autoencoder parameter to support features from the autoencoder
    def __init__(self, device="cuda", global_autoencoder=None):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.debug = True
        # [ADDED] Store the global autoencoder for feature extraction
        self.global_autoencoder = global_autoencoder
        self.device = next(self.global_autoencoder.parameters()).device
        
        # Initialize KMeans first, then load model and distribution
        self.kmeans = None
        _, self.global_class_distribution = self._load_ssl_model_and_distribution()

        print("[SSLEntropy] Initialized with global distribution:")
        for i in range(10):
            print(f"  Class {i}: {self.global_class_distribution[i]:.4f}")
        print(f"[SSLEntropy] Using device: {self.device}")
        
        # For tracking client progress across cycles
        self.client_cycles = {}
        self.client_labeled_sets = {}

    def _load_ssl_model_and_distribution(self):
        if self.global_autoencoder is None:
            raise ValueError("[SSLEntropy] Error: global_autoencoder must be provided!")

        print("[SSLEntropy] Estimating global class distribution using provided global autoencoder")

        # Put model in eval mode
        self.global_autoencoder.eval()

        # Prepare data
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        test_dataset = CIFAR10(dataset_dir, train=False, download=True, transform=test_transform)

        batch_size = 100
        num_samples = min(1000, len(test_dataset))
        indices = torch.randperm(len(test_dataset))[:num_samples]

        features_list = []
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            batch_data = [test_dataset[idx][0] for idx in batch_indices]
            batch_inputs = torch.stack(batch_data).to(self.device)

            with torch.no_grad():
                batch_features = self.global_autoencoder.encode(batch_inputs)
            features_list.append(batch_features.cpu().numpy())

        features = np.vstack(features_list)

        print("[SSLEntropy] Running KMeans clustering to estimate distribution")
        self.kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
        cluster_assignments = self.kmeans.fit_predict(features)

        counts = np.bincount(cluster_assignments, minlength=10)
        distribution = counts / counts.sum()
        global_distribution = {i: float(distribution[i]) for i in range(10)}

        print("[SSLEntropy] Estimated global pseudo-class distribution:")
        for i in range(10):
            print(f"  Pseudo-class {i}: {global_distribution[i]:.4f}")

        return self.global_autoencoder, global_distribution
    
    def compute_target_counts(self, current_distribution, num_samples, labeled_set_size, available_classes):
        """
        Compute the target number of samples to select from each class with advanced balancing.
        
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
            print(f"[SSLEntropy] Planning to select {num_samples} samples")
            print(f"[SSLEntropy] Future labeled set size will be: {future_size}")
            print(f"[SSLEntropy] Available classes in unlabeled pool: {sorted(list(available_classes))}")  
        
        # Check the current representation of available classes
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
                    percentage = 0.0  # 0% for well-represented classes
                
                allocation = min(remaining, int(np.ceil(remaining * percentage)))
                target_counts[cls] = allocation
                remaining -= allocation
        
        # If there's still remaining budget, distribute it to maximize balance
        remaining = num_samples - sum(target_counts.values())
        if remaining > 0:
            if self.debug:
                print(f"[SSLEntropy] Distributing {remaining} remaining samples to maximize balance")
            
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
            print(f"[SSLEntropy] WARNING: Target count {final_count} doesn't match budget {num_samples}")
            
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
            print(f"[SSLEntropy] Target counts per class: {target_counts}")
            print(f"[SSLEntropy] Total samples to select: {sum(target_counts.values())}")
        
        return target_counts

    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, num_samples, labeled_set=None, seed=None):
        print("[SSLEntropy] Selecting samples using pseudo-labels and global distribution with advanced balancing")

        if self.debug:
            print(f"\n[SSLEntropy] Client {client_id}: Selecting {num_samples} samples")
            print(f"[SSLEntropy] Unlabeled pool size: {len(unlabeled_set)}")
            
        # Track current cycle for this client
        if not hasattr(self, 'client_cycles'):
            self.client_cycles = {}
        self.client_cycles[client_id] = self.client_cycles.get(client_id, 0) + 1
        
        # For tracking selections across cycles
        if not hasattr(self, 'client_labeled_sets'):
            self.client_labeled_sets = {}
        
        if client_id not in self.client_labeled_sets:
            self.client_labeled_sets[client_id] = []
            
        # Use the labeled set to track previously selected samples
        total_labeled = 0
        labeled_pseudo_counts = {i: 0 for i in range(10)}
        
        # Extract features and pseudo-labels for all unlabeled samples
        model.eval()
        # [ADDED] Put the autoencoder in eval mode if it exists
        if self.global_autoencoder is not None:
            self.global_autoencoder.eval()
            
        features = []
        indices = []
        entropy_scores = []


        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
                inputs = inputs.to(self.device)
                
                # [ADDED] Use features from the global autoencoder if available, otherwise use SSL model
                if self.global_autoencoder is not None and USE_GLOBAL_SSL:
                    batch_features = self.global_autoencoder.encode(inputs)
                    if self.debug and batch_idx == 0:
                        print(f"[SSLEntropy] Using global autoencoder features (dim={batch_features.size(1)})")
                #else:
                 #   batch_features = self.ssl_model.get_features(inputs)
                  #  if self.debug and batch_idx == 0:
                   #     print(f"[SSLEntropy] Using SSL model features (dim={batch_features.size(1)})")
                        
                features.append(batch_features.cpu().numpy())

                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                log_probs = F.log_softmax(outputs, dim=1)
                probs = torch.exp(log_probs)
                batch_entropy = -torch.sum(probs * log_probs, dim=1)
                entropy_scores.extend(batch_entropy.cpu().numpy())

                batch_indices = unlabeled_loader.sampler.indices[
                    batch_idx * unlabeled_loader.batch_size:
                    min((batch_idx + 1) * unlabeled_loader.batch_size, len(unlabeled_loader.sampler))
                ]
                indices.extend(batch_indices)

        features = np.vstack(features)
        entropy_scores = np.array(entropy_scores)
        
        # Assign pseudo-labels to unlabeled samples
        cluster_assignments = self.kmeans.predict(features)
        pseudo_labels = cluster_assignments
        
        # If labeled set exists, extract its pseudo-labels to calculate current distribution
        if labeled_set is not None and len(labeled_set) > 0:
            total_labeled = len(labeled_set)
            
            # Create a small dataloader for the labeled set to extract features
            from torch.utils.data import DataLoader
            from data.sampler import SubsetSequentialSampler
            
            # Use the same dataset as unlabeled_loader but with the labeled indices
            labeled_loader = DataLoader(
                unlabeled_loader.dataset,
                batch_size=unlabeled_loader.batch_size,
                sampler=SubsetSequentialSampler(labeled_set),
                num_workers=0
            )
            
            # Extract features for labeled samples
            labeled_features = []
            with torch.no_grad():
                for inputs, _ in labeled_loader:
                    inputs = inputs.to(self.device)
                    batch_features = self.global_autoencoder.get_features(inputs)
                    labeled_features.append(batch_features.cpu().numpy())
            
            if labeled_features:
                labeled_features = np.vstack(labeled_features)
                labeled_pseudo_labels = self.kmeans.predict(labeled_features)
                
                # Count by pseudo-label
                for label in labeled_pseudo_labels:
                    labeled_pseudo_counts[label] += 1
        
        # Calculate current distribution from labeled pseudo-labels
        current_distribution = {cls: count/total_labeled for cls, count in labeled_pseudo_counts.items()} if total_labeled > 0 else {cls: 0 for cls in range(10)}
        
        # Print current distribution of labeled data
        print(f"[SSLEntropy] Current pseudo-class distribution:")
        for cls in range(10):
            percentage = current_distribution.get(cls, 0) * 100
            print(f"    Pseudo-class {cls}: {labeled_pseudo_counts.get(cls, 0)} samples ({percentage:.1f}%)")
        
        # Organize unlabeled samples by pseudo-class with their entropy scores
        class_to_samples = {i: [] for i in range(10)}
        for idx, label, entropy in zip(indices, pseudo_labels, entropy_scores):
            class_to_samples[label].append((idx, entropy))
        
        # Get available classes in the unlabeled pool
        available_classes = set(pseudo_labels)
        
        # Count available samples by class
        available_by_class = {cls: len(samples) for cls, samples in class_to_samples.items() if cls in available_classes}
        print(f"[SSLEntropy] Available unlabeled samples by pseudo-class: {available_by_class}")
        
        # Calculate the target counts using the advanced balancing method
        target_counts = self.compute_target_counts(
            current_distribution, 
            num_samples, 
            total_labeled, 
            available_classes
        )
        
        print(f"[SSLEntropy] Target counts to add by pseudo-class: {target_counts}")
        
        # Check if target counts are achievable
        for cls, count in target_counts.items():
            available = available_by_class.get(cls, 0)
            if count > available:
                print(f"[SSLEntropy] Warning: Target count {count} for pseudo-class {cls} exceeds available {available}")
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
        
        # Select samples based on entropy within each pseudo-class
        selected_samples = []
        balanced_selections = {}
        
        # Process classes in order of target count (highest first) for logging clarity
        sorted_classes = sorted(target_counts.keys(), key=lambda cls: target_counts.get(cls, 0), reverse=True)
        
        for cls in sorted_classes:
            if target_counts[cls] > 0:
                if cls not in class_to_samples or len(class_to_samples[cls]) == 0:
                    print(f"[SSLEntropy] No unlabeled samples available for pseudo-class {cls}")
                    continue
                
                # Sort samples by entropy (highest first)
                class_samples = class_to_samples[cls]
                class_samples.sort(key=lambda x: x[1], reverse=True)
                
                # Select top samples by entropy
                num_to_select = min(target_counts[cls], len(class_samples))
                selected_indices = [sample[0] for sample in class_samples[:num_to_select]]
                selected_samples.extend(selected_indices)
                
                balanced_selections[cls] = num_to_select
                print(f"[SSLEntropy] Selected {num_to_select} samples from pseudo-class {cls}")
        
        # Calculate how much of the budget went to balanced selection
        balanced_count = len(selected_samples)
        balanced_percentage = balanced_count / num_samples * 100 if num_samples > 0 else 0
        print(f"[SSLEntropy] Selected {balanced_count}/{num_samples} samples ({balanced_percentage:.1f}%) using class balancing")
        print(f"[SSLEntropy] Class-balanced selections: {balanced_selections}")
        
        # Double-check: Are we missing any selection?
        remaining_to_select = num_samples - len(selected_samples)
        if remaining_to_select > 0:
            print(f"[SSLEntropy] WARNING: Still need to select {remaining_to_select} samples")
            print(f"[SSLEntropy] This should not happen with proper budget allocation!")
            
            # As a failsafe, select additional samples from underrepresented classes
            # We'll prioritize by representation ratio rather than just entropy
            remaining_samples = []
            for i, idx in enumerate(unlabeled_set):
                if idx not in selected_samples and i < len(pseudo_labels):
                    label = pseudo_labels[i]
                    ratio = current_distribution.get(label, 0) / self.global_class_distribution[label] \
                        if self.global_class_distribution[label] > 0 else float('inf')
                    remaining_samples.append((idx, entropy_scores[i], label, ratio))
            
            # Sort by representation ratio (lowest first), then by entropy (highest first)
            remaining_samples.sort(key=lambda x: (x[3], -x[1]))
            
            # Select additional samples
            additional_samples = [sample[0] for sample in remaining_samples[:remaining_to_select]]
            selected_samples.extend(additional_samples)
            
            additional_labels = [sample[2] for sample in remaining_samples[:remaining_to_select]]
            additional_by_class = {}
            for cls in range(10):
                additional_by_class[cls] = sum(1 for label in additional_labels if label == cls)
                
            print(f"[SSLEntropy] FAILSAFE - Additional samples by pseudo-class: {additional_by_class}")
            print(f"[SSLEntropy] Selected {len(additional_samples)} additional samples from underrepresented classes")
            
        # Calculate remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        # Debug info about final selection
        selected_pseudo_classes = [pseudo_labels[indices.index(idx)] if idx in indices else -1 for idx in selected_samples]
        final_class_counts = {}
        for cls in range(10):
            final_class_counts[cls] = sum(1 for label in selected_pseudo_classes if label == cls)
        
        print(f"\n[SSLEntropy] Final selection pseudo-class distribution:")
        for cls in range(10):
            count = final_class_counts.get(cls, 0)
            percentage = count / len(selected_samples) * 100 if len(selected_samples) > 0 else 0
            target = target_counts.get(cls, 0)
            print(f"    Pseudo-class {cls}: {count} samples ({percentage:.1f}%) [Target: {target}]")
            
        print(f"[SSLEntropy] Total selected: {len(selected_samples)} out of budget {num_samples}")
        
        # Track selections for reference
        self.client_labeled_sets[client_id] = self.client_labeled_sets.get(client_id, []) + selected_samples
        
        # Calculate the new distribution after this selection
        future_distribution = {}
        future_size = total_labeled + len(selected_samples)
        for cls in range(10):
            future_distribution[cls] = (labeled_pseudo_counts.get(cls, 0) + final_class_counts.get(cls, 0)) / future_size
        
        # Calculate how close we got to the global distribution
        dist_error = sum(abs(future_distribution.get(cls, 0) - self.global_class_distribution[cls]) for cls in range(10)) / 2
        print(f"[SSLEntropy] Distribution error after selection: {dist_error:.4f} (lower is better)")
        
        # Display improvement compared to initial distribution
        if total_labeled > 0:
            initial_distribution = {cls: count/total_labeled for cls, count in labeled_pseudo_counts.items()}
            initial_error = sum(abs(initial_distribution.get(cls, 0) - self.global_class_distribution[cls]) for cls in range(10)) / 2
            improvement = initial_error - dist_error
            print(f"[SSLEntropy] Initial error: {initial_error:.4f}, Improvement: {improvement:.4f} ({improvement/initial_error*100:.1f}%)")
        
        return selected_samples, remaining_unlabeled