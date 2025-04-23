import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from models.ssl.contrastive_model import SimpleContrastiveLearning
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
from sklearn.cluster import KMeans

# [ADDED] Import config to access global SSL settings
from config import USE_GLOBAL_SSL

class SSLEntropySampler:
    def __init__(self, device="cuda", global_autoencoder=None):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.debug = True
        
        # Load SimCLR model from checkpoint
        print("[SSLEntropy] Loading SimCLR model from checkpoint")
        self.ssl_model = self._load_ssl_model_from_checkpoint()
        
        # Initialize class centroids and distribution
        self.class_centroids = None
        self.global_class_distribution = self._estimate_global_distribution()
        print(f"[SSLEntropy] Using device: {self.device}")

        
        # For tracking client progress across cycles
        self.client_cycles = {}
        self.client_labeled_sets = {}

    def _load_ssl_model_from_checkpoint(self):
        """Load SimCLR model from checkpoint"""
        # Define checkpoint path
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SSL_checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pt')
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"[SSLEntropy] Error: SSL checkpoint not found at {checkpoint_path}")
        
        # Initialize SimCLR model
        model = SimpleContrastiveLearning(feature_dim=128, temperature=0.5)
        model = model.to(self.device)
        
        # Load checkpoint
        print(f"[SSLEntropy] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[SSLEntropy] Loaded SimCLR model from round {checkpoint['round']}")
        
        model.eval()
        
        return model

    def _estimate_global_distribution(self):
        """Estimate global class distribution using the loaded SimCLR model"""
        if self.ssl_model is None:
            raise ValueError("[SSLEntropy] Error: SSL model must be loaded first!")
        
        print("[SSLEntropy] Estimating global class distribution using SimCLR model")
        
        # Prepare data
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        test_dataset = CIFAR10(dataset_dir, train=False, download=True, transform=test_transform)
        
        batch_size = 100
        num_samples = min(10000, len(test_dataset))  
        indices = torch.randperm(len(test_dataset))[:num_samples]
        
        features_list = []
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            batch_data = [test_dataset[idx][0] for idx in batch_indices]
            batch_inputs = torch.stack(batch_data).to(self.device)
            
            with torch.no_grad():
                # Use get_features method for SimCLR model
                batch_features = self.ssl_model.get_features(batch_inputs)
            
            features_list.append(batch_features.cpu().numpy())
        
        features = np.vstack(features_list)
        
        print(f"[SSLEntropy] Running KMeans clustering on {num_samples} samples to estimate distribution")
        from sklearn.cluster import KMeans
        self.kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto")
        cluster_assignments = self.kmeans.fit_predict(features)
        
        counts = np.bincount(cluster_assignments, minlength=10)
        distribution = counts / counts.sum()
        
        global_distribution = {i: float(distribution[i]) for i in range(10)}
        
        print("[SSLEntropy] Estimated global pseudo-class distribution:")
        for i in range(10):
            print(f"  Pseudo-class {i}: {global_distribution[i]:.4f}")
        
        return global_distribution

    def compute_class_centroids(self, model, labeled_loader):
        """
        Compute class centroids from labeled data using the SSL model features
        
        Args:
            model: The current model
            labeled_loader: DataLoader for labeled samples
            
        Returns:
            dict: Class centroids mapping class -> centroid vector
        """
        print("[SSLEntropy] Computing class centroids from labeled data")
        
        model_device = next(model.parameters()).device
        self.ssl_model = self.ssl_model.to(model_device)
        self.ssl_model.eval()
        
        # Store embeddings by class
        class_embeddings = {}
        
        with torch.no_grad():
            for inputs, targets in labeled_loader:
                inputs = inputs.to(model_device)
                
                # Extract features using the SSL model
                batch_features = self.ssl_model.get_features(inputs)
                batch_features = batch_features.cpu().numpy()
                
                # Group by class
                for i, target in enumerate(targets.numpy()):
                    if target not in class_embeddings:
                        class_embeddings[target] = []
                    class_embeddings[target].append(batch_features[i])
        
        # Compute centroids for each class
        centroids = {}
        for cls, embeddings in class_embeddings.items():
            if embeddings:  # Ensure we have embeddings for this class
                centroids[cls] = np.mean(np.stack(embeddings), axis=0)
                
        if self.debug:
            print(f"[SSLEntropy] Computed centroids for {len(centroids)} classes")
        
        return centroids
    
    def extract_features(self, model, data_loader):
        """
        Extract features from data using the SSL model
        
        Args:
            model: The current model
            data_loader: DataLoader for data samples
            
        Returns:
            tuple: (features, indices, entropy_scores)
        """
        model_device = next(model.parameters()).device
        self.ssl_model = self.ssl_model.to(model_device)
        
        features = []
        indices = []
        entropy_scores = []

        model.eval()
        self.ssl_model.eval()
        
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(data_loader):
                inputs = inputs.to(model_device)
                
                # Use features from the SimCLR model
                batch_features = self.ssl_model.get_features(inputs)
                if self.debug and batch_idx == 0:
                    print(f"[SSLEntropy] Using SimCLR model features (dim={batch_features.size(1)})")
                        
                features.append(batch_features.cpu().numpy())

                # Calculate entropy from the current model
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                log_probs = F.log_softmax(outputs, dim=1)
                probs = torch.exp(log_probs)
                batch_entropy = -torch.sum(probs * log_probs, dim=1)
                entropy_scores.extend(batch_entropy.cpu().numpy())

                batch_indices = data_loader.sampler.indices[
                    batch_idx * data_loader.batch_size:
                    min((batch_idx + 1) * data_loader.batch_size, len(data_loader.sampler))
                ]
                indices.extend(batch_indices)

        features = np.vstack(features)
        entropy_scores = np.array(entropy_scores)
        
        return features, indices, entropy_scores
    
    def assign_pseudo_labels(self, features, class_centroids):
        """
        Assign pseudo-labels to samples based on nearest class centroid
        
        Args:
            features: numpy array of feature vectors
            class_centroids: dict mapping class -> centroid vector
            
        Returns:
            tuple: (pseudo_labels, confidence_scores)
        """
        if not class_centroids or len(class_centroids) == 0:
            print("[SSLEntropy] Error: No class centroids available for pseudo-labeling")
            return None, None
            
        # Stack centroids into a matrix for efficient computation
        centroid_classes = sorted(class_centroids.keys())
        centroid_matrix = np.stack([class_centroids[c] for c in centroid_classes])
        
        # Compute cosine similarity between features and centroids
        similarity = cosine_similarity(features, centroid_matrix)
        
        # Get the most similar centroid for each sample
        most_similar_idx = np.argmax(similarity, axis=1)
        pseudo_labels = np.array([centroid_classes[idx] for idx in most_similar_idx])
        
        # Use the similarity score as confidence
        confidence_scores = np.max(similarity, axis=1)
        
        if self.debug:
            print(f"[SSLEntropy] Assigned pseudo-labels to {len(pseudo_labels)} samples")
                
        return pseudo_labels, confidence_scores
    
    def compute_target_counts(self, current_distribution, num_samples, labeled_set_size, available_classes):
        """
        Compute the target number of samples to select from each class with balancing.
        
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
            print(f"[SSLEntropy] Planning to select {num_samples} samples from {len(available_classes)} available classes")
        
        # Calculate representation ratios for available classes
        # This measures how well each class is represented compared to its target
        representation_ratios = {}
        missing_classes = []
        
        for cls in available_classes:
            target_global_ratio = self.global_class_distribution[cls]
            
            # Calculate representation ratio
            if labeled_set_size > 0:
                current_ratio = current_distribution.get(cls, 0)
                ratio = current_ratio / target_global_ratio if target_global_ratio > 0 else float('inf')
                representation_ratios[cls] = ratio
                
                # Identify missing classes (0 samples)
                if current_ratio == 0:
                    missing_classes.append(cls)
            else:
                # If no labeled samples yet, all classes are equally unrepresented
                representation_ratios[cls] = 0.0
        
        # First prioritize missing classes
        if missing_classes and labeled_set_size > 0:
            # Allocate some samples to each missing class
            samples_per_missing = max(1, min(5, num_samples // len(missing_classes)))
            for cls in missing_classes:
                target_counts[cls] = samples_per_missing
            
            remaining = num_samples - sum(target_counts.values())
        else:
            remaining = num_samples
        
        # Then distribute remaining samples inversely proportional to representation
        if remaining > 0:
            # Calculate inverse ratios (lower ratio = higher priority)
            inverse_ratios = {}
            total_inverse = 0
            
            for cls in available_classes:
                ratio = representation_ratios.get(cls, 0)
                # Adding a small constant to avoid division by zero
                inverse = 1.0 / (ratio + 0.01) if ratio > 0 else 100
                inverse_ratios[cls] = inverse
                total_inverse += inverse
            
            # Distribute remaining samples proportionally to inverse ratios
            allocated = 0
            for cls in sorted(available_classes, key=lambda c: representation_ratios.get(c, float('inf'))):
                if total_inverse > 0:
                    to_allocate = int(np.floor(remaining * inverse_ratios[cls] / total_inverse))
                    # Ensure we allocate at least 1 sample to each class if possible
                    to_allocate = max(1, to_allocate) if remaining - allocated >= len(available_classes) - len(target_counts) else to_allocate
                    target_counts[cls] = target_counts.get(cls, 0) + to_allocate
                    allocated += to_allocate
            
            # If we still have samples left, distribute one by one to the most underrepresented classes
            remaining_after_allocation = num_samples - sum(target_counts.values())
            if remaining_after_allocation > 0:
                sorted_classes = sorted(available_classes, key=lambda c: representation_ratios.get(c, float('inf')))
                idx = 0
                while remaining_after_allocation > 0 and idx < len(sorted_classes):
                    cls = sorted_classes[idx]
                    target_counts[cls] = target_counts.get(cls, 0) + 1
                    remaining_after_allocation -= 1
                    idx = (idx + 1) % len(sorted_classes)
        
        if self.debug:
            print(f"[SSLEntropy] Target counts per class: {target_counts}")
            print(f"[SSLEntropy] Target selection: {sum(target_counts.values())} samples across {len(target_counts)} classes")
        
        return target_counts
        
    def select_samples(self, model, model_server, unlabeled_loader, client_id, unlabeled_set, num_samples, labeled_set=None, seed=None):
        print("[SSLEntropy] Selecting samples using centroid-based pseudo-labels and global distribution")

        if self.debug:
            print(f"\n[SSLEntropy] Client {client_id}: Selecting {num_samples} samples")
            print(f"[SSLEntropy] Unlabeled pool size: {len(unlabeled_set)}")
            if labeled_set is not None:
                print(f"[SSLEntropy] Labeled set size: {len(labeled_set)}")
            else:
                print("[SSLEntropy] No labeled set provided")
        
        # Move SSL model to match client model device
        model_device = next(model.parameters()).device
        if next(self.ssl_model.parameters()).device != model_device:
            print(f"[SSLEntropy] Moving SSL model from {next(self.ssl_model.parameters()).device} to {model_device}")
            self.ssl_model = self.ssl_model.to(model_device)
            
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
        labeled_class_counts = {i: 0 for i in range(10)}
        
        # Step 1: If we have labeled data, compute class centroids
        class_centroids = None
        if labeled_set is not None and len(labeled_set) > 0:
            total_labeled = len(labeled_set)
            
            # Create a dataloader for the labeled set
            labeled_loader = DataLoader(
                unlabeled_loader.dataset,
                batch_size=unlabeled_loader.batch_size,
                sampler=SubsetSequentialSampler(labeled_set),
                num_workers=0
            )
            
            # Get actual labels to compute current distribution
            labeled_classes = []
            for _, targets in labeled_loader:
                labeled_classes.extend(targets.numpy())
            
            # Count by class
            for label in labeled_classes:
                labeled_class_counts[label] += 1
                
            # Compute class centroids from labeled data
            class_centroids = self.compute_class_centroids(model, labeled_loader)
            
            # Print current distribution of labeled data
            print(f"[SSLEntropy] Current class distribution in labeled set:")
            for cls in range(10):
                percentage = (labeled_class_counts.get(cls, 0) / total_labeled * 100) if total_labeled > 0 else 0
                print(f"  Class {cls}: {labeled_class_counts.get(cls, 0)} samples ({percentage:.1f}%)")
            
            # Summarize current distribution of labeled data
            print(f"[SSLEntropy] Current labeled set: {total_labeled} samples across {len([c for c, count in labeled_class_counts.items() if count > 0])} classes")
                
            # Check if we have centroids for all classes
            missing_centroids = [cls for cls in range(10) if cls not in class_centroids]
            if missing_centroids:
                print(f"[SSLEntropy] Warning: Missing centroids for classes: {missing_centroids}")
        else:
            print("[SSLEntropy] No labeled data available to compute class centroids")
            
        # Step 2: Extract features from unlabeled data
        print("[SSLEntropy] Extracting features from unlabeled data")
        features, indices, entropy_scores = self.extract_features(model, unlabeled_loader)
        print(f"[SSLEntropy] Extracted features for {len(features)} unlabeled samples")
        
        # Step 3: Assign pseudo-labels using class centroids
        pseudo_labels = None
        confidence_scores = None
        
        if class_centroids and len(class_centroids.keys()) > 0:
            # Use centroid-based pseudo-labeling
            print("[SSLEntropy] Assigning pseudo-labels using class centroids")
            pseudo_labels, confidence_scores = self.assign_pseudo_labels(features, class_centroids)
            if pseudo_labels is None:
                print("[SSLEntropy] Failed to assign pseudo-labels, falling back to model predictions")
                # Fall back to model predictions
                pseudo_labels, confidence_scores = self._get_model_predictions(model, unlabeled_loader)
            else:
                print(f"[SSLEntropy] Successfully assigned centroid-based pseudo-labels to {len(pseudo_labels)} samples")
        else:
            # If we don't have centroids, use model predictions
            print("[SSLEntropy] No class centroids available, using model predictions as pseudo-labels")
            pseudo_labels, confidence_scores = self._get_model_predictions(model, unlabeled_loader)
            
        # Calculate current distribution from labeled data
        current_distribution = {cls: count/total_labeled for cls, count in labeled_class_counts.items()} if total_labeled > 0 else {i: 0 for i in range(10)}
        
        # Step 4: Select samples based on class distribution and confidence
        # Organize samples by pseudo-class
        class_to_samples = {i: [] for i in range(10)}
        for idx, sample_idx, label, entropy, confidence in zip(range(len(indices)), indices, pseudo_labels, entropy_scores, confidence_scores):
            class_to_samples[label].append((sample_idx, entropy, confidence))
        
        # Get available classes in the unlabeled pool
        available_classes = set(pseudo_labels)
        
        # Count available samples by class
        available_by_class = {cls: len(samples) for cls, samples in class_to_samples.items() if cls in available_classes}
        print(f"[SSLEntropy] Available unlabeled samples by pseudo-class: {available_by_class}")
        
        # Calculate target counts for each class
        target_counts = self.compute_target_counts(
            current_distribution, 
            num_samples, 
            total_labeled, 
            available_classes
        )
        
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
                print(f"[SSLEntropy] Redistributing {remaining} samples to classes with extra capacity")
                
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
                    # No need to log each redistribution
                    if remaining <= 0:
                        break
                
                print(f"[SSLEntropy] Updated target counts after redistribution: {target_counts}")
                print(f"[SSLEntropy] After redistribution: {sum(target_counts.values())} samples across {len(target_counts)} classes")
                
        # Select samples from each class based on target counts
        selected_samples = []
        
        # Select samples from each class
        for cls, count in target_counts.items():
            if count > 0 and cls in class_to_samples and class_to_samples[cls]:
                # Sort samples by confidence (highest first)
                samples_in_class = class_to_samples[cls]
                samples_in_class.sort(key=lambda x: x[2], reverse=True)
                
                # Select top confident samples
                to_select = min(count, len(samples_in_class))
                selected_indices = [sample[0] for sample in samples_in_class[:to_select]]
                selected_samples.extend(selected_indices)
                
                # Log selection per class
                print(f"[SSLEntropy] Selected {to_select} samples from pseudo-class {cls} (highest confidence first)")
                
        # Verify we selected the correct number of samples
        if len(selected_samples) < num_samples:
            print(f"[SSLEntropy] Warning: Only selected {len(selected_samples)}/{num_samples} samples")
            # Select additional samples based on entropy if needed
            remaining = num_samples - len(selected_samples)
            if remaining > 0:
                # Create a list of (index, entropy) for unselected samples
                unselected = [(idx, entropy_scores[i]) for i, idx in enumerate(indices) 
                            if idx not in selected_samples]
                
                # Sort by entropy (highest first)
                unselected.sort(key=lambda x: x[1], reverse=True)
                
                # Select additional samples
                additional = [item[0] for item in unselected[:remaining]]
                selected_samples.extend(additional)
                print(f"[SSLEntropy] Selected {len(additional)} additional samples based on entropy")
        
        # Calculate remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        # Track selections
        self.client_labeled_sets[client_id].extend(selected_samples)
        
        # Add detailed class summary to final selection report
        selected_class_counts = {}
        for idx in selected_samples:
            if idx in indices:
                selected_class = pseudo_labels[indices.index(idx)]
                selected_class_counts[selected_class] = selected_class_counts.get(selected_class, 0) + 1
                
        print(f"[SSLEntropy] Final selection by pseudo-class:")
        for cls in sorted(selected_class_counts.keys()):
            count = selected_class_counts[cls]
            percentage = count / len(selected_samples) * 100
            print(f"  Pseudo-class {cls}: {count} samples ({percentage:.1f}%)")
        
        print(f"[SSLEntropy] Final selection: {len(selected_samples)} samples across {len(selected_class_counts)} classes")
        
        return selected_samples, remaining_unlabeled
        
    def _get_model_predictions(self, model, data_loader):
        """
        Get model predictions as pseudo-labels
        
        Args:
            model: The current model
            data_loader: DataLoader for data samples
            
        Returns:
            tuple: (pseudo_labels, confidence_scores)
        """
        model_device = next(model.parameters()).device
        
        all_probs = []
        model.eval()
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(model_device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                
        all_probs = np.vstack(all_probs)
        pseudo_labels = np.argmax(all_probs, axis=1)
        confidence_scores = np.max(all_probs, axis=1)
        
        return pseudo_labels, confidence_scores