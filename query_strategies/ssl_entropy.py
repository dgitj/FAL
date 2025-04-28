import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from models.ssl.resnet50_contrastive import ContrastiveModel
from models.ssl.contrastive_model import SimpleContrastiveLearning
from models.ssl.model_utils import load_checkpoint_flexible

# Import custom model creator
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#from create_matching_model import CustomContrastiveModel

from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
from sklearn.cluster import KMeans

# [ADDED] Import config to access global SSL settings
from config import USE_GLOBAL_SSL

class SSLEntropySampler:
    def __init__(self, device="cuda", global_autoencoder=None, distance_threshold=0.3, confidence_threshold=0.7):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.debug = True
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        
        print(f"[SSLEntropy] Using distance_threshold={distance_threshold}, confidence_threshold={confidence_threshold}")
        
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
        
        # Check if we should use ResNet50 model
        checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pt')
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"[SSLEntropy] Error: SSL checkpoint not found at {checkpoint_path}")
        
        # Log checkpoint information for debugging
        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
        checkpoint_modified = os.path.getmtime(checkpoint_path)
        import datetime
        modified_time = datetime.datetime.fromtimestamp(checkpoint_modified).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[SSLEntropy] Loading checkpoint: {checkpoint_path}")
        print(f"[SSLEntropy] Checkpoint size: {checkpoint_size:.2f} MB")
        print(f"[SSLEntropy] Last modified: {modified_time}")
        
        # Calculate checkpoint hash for verification
        import hashlib
        with open(checkpoint_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        print(f"[SSLEntropy] Checkpoint MD5 hash: {file_hash}")
        
        # First, load a small part of the checkpoint to determine its structure
        try:
            # Load just metadata or a small portion to analyze structure
            checkpoint_sample = torch.load(checkpoint_path, map_location=self.device)
            # Check if it looks like a ResNet50 checkpoint by looking at key patterns
            is_resnet50 = False
            if isinstance(checkpoint_sample, dict):
                # Check a few keys that would indicate ResNet50
                key_list = list(checkpoint_sample.keys())
                if any('layer4' in k for k in key_list) or \
                   any('layer3' in k for k in key_list) or \
                   any('encoder.layer' in k for k in key_list):
                    is_resnet50 = True
                    print(f"[SSLEntropy] Detected ResNet50 checkpoint structure")
                else:
                    print(f"[SSLEntropy] Detected standard encoder structure")
            
            # Initialize the appropriate model based on detected structure
            print(f"[SSLEntropy] Creating custom model to match checkpoint structure")
            model = CustomContrastiveModel(checkpoint_sample)
            
            model = model.to(self.device)
            print(f"[SSLEntropy] Successfully created matching model")
            
            # Calculate model parameter hash for verification
            model_hash = hash(str(sum(p.sum().item() for p in model.parameters())))
            print(f"[SSLEntropy] Model parameter hash: {model_hash}")
            
            # No need to manually load state dict as it was already loaded during model creation
            print(f"[SSLEntropy] Checkpoint loaded during model creation")
                
        except Exception as e:
            print(f"[SSLEntropy] Error setting up model: {str(e)}")
            raise
        
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
        num_samples = min(5000, len(test_dataset))  
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
        Compute class centroids from labeled data using the SSL model features with L2 normalization
        
        Args:
            model: The current model
            labeled_loader: DataLoader for labeled samples
            
        Returns:
            dict: Class centroids mapping class -> centroid vector
        """
        print("[SSLEntropy] Computing class centroids from labeled data with L2 normalization")
        
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
                
                # Apply L2 normalization to embeddings
                for i in range(len(batch_features)):
                    batch_features[i] = batch_features[i] / np.linalg.norm(batch_features[i])
                
                # Group by class
                for i, target in enumerate(targets.numpy()):
                    if target not in class_embeddings:
                        class_embeddings[target] = []
                    class_embeddings[target].append(batch_features[i])
        
        # Compute centroids for each class
        centroids = {}
        for cls, embeddings in class_embeddings.items():
            if embeddings:  # Ensure we have embeddings for this class
                # Note: Embeddings are already L2 normalized
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
                
                # Convert to numpy and normalize
                np_features = batch_features.cpu().numpy()
                for i in range(len(np_features)):
                    np_features[i] = np_features[i] / np.linalg.norm(np_features[i])
                        
                features.append(np_features)

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
        Assign pseudo-labels to samples based on nearest class centroid, applying
        both distance and confidence thresholds
        
        Args:
            features: numpy array of feature vectors
            class_centroids: dict mapping class -> centroid vector
            
        Returns:
            tuple: (pseudo_labels, confidence_scores, rejected_indices)
        """
        if not class_centroids or len(class_centroids) == 0:
            print("[SSLEntropy] Error: No class centroids available for pseudo-labeling")
            return None, None, None
            
        # Stack centroids into a matrix for efficient computation
        centroid_classes = sorted(class_centroids.keys())
        centroid_matrix = np.stack([class_centroids[c] for c in centroid_classes])
        
        # Normalize the centroids (they may not be normalized if computed as means)
        for i in range(len(centroid_matrix)):
            centroid_matrix[i] = centroid_matrix[i] / np.linalg.norm(centroid_matrix[i])
        
        # Compute cosine similarity between features and centroids
        similarity = cosine_similarity(features, centroid_matrix)
        
        # Initialize arrays for pseudo-labels and confidence scores
        pseudo_labels = np.zeros(len(features), dtype=int)
        confidence_scores = np.zeros(len(features))
        rejected_mask = np.zeros(len(features), dtype=bool)
        
        # For each sample, apply distance and confidence thresholds
        for i in range(len(features)):
            # Get maximum similarity score (distance check)
            max_similarity = np.max(similarity[i])
            
            # Check if the sample passes the distance threshold
            if max_similarity < self.distance_threshold:
                # Sample doesn't fit anywhere in the embedding space
                rejected_mask[i] = True
                continue
                
            # Get the most similar centroid
            most_similar_idx = np.argmax(similarity[i])
            max_confidence = similarity[i][most_similar_idx]
            
            # Check if the confidence exceeds the threshold
            if max_confidence < self.confidence_threshold:
                # Sample isn't confidently assigned to any class
                rejected_mask[i] = True
                continue
                
            # If we got here, the sample passed both checks
            pseudo_labels[i] = centroid_classes[most_similar_idx]
            confidence_scores[i] = max_confidence
        
        # Get indices of accepted and rejected samples
        accepted_indices = np.where(~rejected_mask)[0]
        rejected_indices = np.where(rejected_mask)[0]
        
        # Only keep labels and scores for accepted samples
        accepted_pseudo_labels = pseudo_labels[accepted_indices]
        accepted_confidence_scores = confidence_scores[accepted_indices]
        
        if self.debug:
            num_accepted = len(accepted_indices)
            num_rejected = len(rejected_indices)
            total = len(features)
            print(f"[SSLEntropy] Assigned pseudo-labels to {num_accepted}/{total} samples ({num_accepted/total*100:.1f}%)")
            print(f"[SSLEntropy] Rejected {num_rejected}/{total} samples ({num_rejected/total*100:.1f}%)")
            print(f"[SSLEntropy] Reasons: distance threshold={self.distance_threshold}, confidence threshold={self.confidence_threshold}")
            
            # Count accepted samples by class
            if num_accepted > 0:
                accepted_counts = {}
                for cls in np.unique(accepted_pseudo_labels):
                    count = np.sum(accepted_pseudo_labels == cls)
                    accepted_counts[cls] = count
                    
                print(f"[SSLEntropy] Accepted samples by class: {accepted_counts}")
                
        # Convert to lists of original indices
        filtered_indices = [accepted_indices[i] for i in range(len(accepted_indices))]
        filtered_pseudo_labels = [accepted_pseudo_labels[i] for i in range(len(accepted_pseudo_labels))]
        filtered_confidence_scores = [accepted_confidence_scores[i] for i in range(len(accepted_confidence_scores))]
                
        return filtered_pseudo_labels, filtered_confidence_scores, rejected_indices
    
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

        # If we have labeled data, visualize it first
        if labeled_set is not None and len(labeled_set) > 0:
            print("[SSLEntropy] Visualizing labeled data embeddings...")
            try:
                # Create a separate dataloader for visualization
                vis_labeled_loader = DataLoader(
                    unlabeled_loader.dataset,
                    batch_size=unlabeled_loader.batch_size,
                    sampler=SubsetSequentialSampler(labeled_set),
                    num_workers=0
                )
                
                # Ensure output directory exists
                import os
                output_dir = os.path.abspath('ssl_visualizations')
                os.makedirs(output_dir, exist_ok=True)
                
                # Visualize the labeled data embeddings
                self.visualize_labeled_embeddings(
                    model,
                    vis_labeled_loader,
                    output_dir=output_dir,
                    client_id=client_id
                )
            except Exception as e:
                print(f"[ERROR] Failed to visualize labeled data: {str(e)}")
                import traceback
                traceback.print_exc()  # This won't interrupt the main algorithm
            
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
        rejected_indices = None
        
        if class_centroids and len(class_centroids.keys()) > 0:
            # Use centroid-based pseudo-labeling
            print("[SSLEntropy] Assigning pseudo-labels using class centroids with thresholds")
            pseudo_labels, confidence_scores, rejected_indices = self.assign_pseudo_labels(features, class_centroids)
            if pseudo_labels is None or len(pseudo_labels) == 0:
                error_msg = "[SSLEntropy] ERROR: Failed to assign any pseudo-labels using class centroids"
                print(error_msg)
                raise ValueError(error_msg)
            else:
                total_samples = len(features)
                num_rejected = len(rejected_indices) if rejected_indices is not None else 0
                print(f"[SSLEntropy] Successfully assigned centroid-based pseudo-labels to {len(pseudo_labels)}/{total_samples} samples")
                print(f"[SSLEntropy] Rejected {num_rejected}/{total_samples} samples that didn't pass thresholds")
        else:
            # If we don't have centroids, raise an error
            error_msg = "[SSLEntropy] ERROR: No class centroids available for pseudo-labeling"
            print(error_msg)
            raise ValueError(error_msg)
            
        # Calculate current distribution from labeled data
        current_distribution = {cls: count/total_labeled for cls, count in labeled_class_counts.items()} if total_labeled > 0 else {i: 0 for i in range(10)}
        
        # Step 4: Select samples based on class distribution and confidence
        # Organize samples by pseudo-class
        class_to_samples = {i: [] for i in range(10)}
        for idx, pl, cs, es in zip(indices, pseudo_labels, confidence_scores, entropy_scores):
            class_to_samples[pl].append((idx, es, cs))
        
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
                # Sort samples by entropy (highest first)
                samples_in_class = class_to_samples[cls]
                samples_in_class.sort(key=lambda x: x[1], reverse=True)
                
                # Select highest entropy samples
                to_select = min(count, len(samples_in_class))
                selected_indices = [sample[0] for sample in samples_in_class[:to_select]]
                selected_samples.extend(selected_indices)
                
                # Log selection per class
                print(f"[SSLEntropy] Selected {to_select} samples from pseudo-class {cls} (highest entropy first)")
                
        # Add fallback for rejected samples if we can't meet the target
        if len(selected_samples) < num_samples and rejected_indices is not None and len(rejected_indices) > 0:
            print(f"[SSLEntropy] Using high-entropy rejected samples as fallback")
            remaining = num_samples - len(selected_samples)
            
            # Get the rejected indices with their entropy scores
            rejected_with_entropy = []
            for i in rejected_indices:
                # Convert numpy int64 to regular int for indexing
                idx = int(i)  
                if idx < len(indices):
                    orig_idx = indices[idx]
                    entropy = entropy_scores[idx]
                    # Only consider samples that haven't been selected yet
                    if orig_idx not in selected_samples:
                        rejected_with_entropy.append((orig_idx, entropy))
            
            # Sort by entropy (highest first)
            rejected_with_entropy.sort(key=lambda x: x[1], reverse=True)
            
            # Select additional samples up to the limit
            to_add = min(remaining, len(rejected_with_entropy))
            if to_add > 0:
                fallback_samples = [x[0] for x in rejected_with_entropy[:to_add]]
                selected_samples.extend(fallback_samples)
                print(f"[SSLEntropy] Added {to_add} high-entropy rejected samples as fallback")
        
        # Calculate remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        # Track selections
        self.client_labeled_sets[client_id].extend(selected_samples)
        
        # Add detailed class summary to final selection report
        selected_class_counts = {}
        
        # Map selected samples to their pseudo-classes
        mapped_count = 0
        for idx in selected_samples:
            found = False
            # Look through our samples with pseudo-labels
            for i, orig_idx in enumerate(indices):
                if orig_idx == idx and i < len(pseudo_labels):
                    pseudo_class = pseudo_labels[i]
                    selected_class_counts[pseudo_class] = selected_class_counts.get(pseudo_class, 0) + 1
                    found = True
                    mapped_count += 1
                    break
                    
        print(f"[SSLEntropy] Mapped {mapped_count}/{len(selected_samples)} selected samples to pseudo-classes")
        
        if mapped_count > 0:
            print(f"[SSLEntropy] Final selection by pseudo-class:")
            for cls in sorted(selected_class_counts.keys()):
                count = selected_class_counts[cls]
                percentage = count / mapped_count * 100
                print(f"  Pseudo-class {cls}: {count} samples ({percentage:.1f}%)")
        
        print(f"[SSLEntropy] Final selection: {len(selected_samples)} samples across {len(selected_class_counts)} classes")
        
        return selected_samples, remaining_unlabeled
        
    def visualize_labeled_embeddings(self, model, labeled_loader, output_dir='labeled_embeddings', client_id=None):
        """
        Visualize embeddings of labeled data using t-SNE
        
        Args:
            model: Current model
            labeled_loader: DataLoader containing labeled samples
            output_dir: Directory to save visualizations
            client_id: Client ID for naming files
        """
        print(f"[DEBUG] Starting visualization for client {client_id} with {len(labeled_loader.sampler)} labeled samples")
        
        try:
            import matplotlib
            # Force non-interactive backend for headless environments
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import os
            
            # Create full output directory path with all parents
            output_dir = os.path.abspath(f'{output_dir}/client_{client_id}')
            os.makedirs(output_dir, exist_ok=True)
            print(f"[DEBUG] Will save visualization to: {output_dir}")
            
            print("[SSLEntropy] Extracting features from labeled data for visualization...")
            
            # Extract features and labels
            features = []
            labels = []
            
            model.eval()
            self.ssl_model.eval()
            
            labeled_count = 0
            with torch.no_grad():
                for inputs, targets in labeled_loader:
                    labeled_count += len(inputs)
                    inputs = inputs.to(self.device)
                    
                    # Extract features using SSL model
                    batch_features = self.ssl_model.get_features(inputs)
                    batch_features = batch_features.cpu().numpy()
                    
                    # Apply L2 normalization
                    for i in range(len(batch_features)):
                        batch_features[i] = batch_features[i] / np.linalg.norm(batch_features[i])
                    
                    features.append(batch_features)
                    labels.extend(targets.numpy())
            
            print(f"[DEBUG] Processed {labeled_count} labeled samples")
            
            # Stack features and convert labels to numpy array
            features = np.vstack(features) if features else np.array([])
            labels = np.array(labels)
            
            if len(features) == 0:
                print("[SSLEntropy] Warning: No labeled data to visualize")
                return
            
            print(f"[DEBUG] Feature shape: {features.shape}, Labels shape: {labels.shape}")
                
            # Apply t-SNE with adjusted perplexity and a random seed based on client_id
            print(f"[SSLEntropy] Running t-SNE on {len(features)} labeled samples...")
            perplexity = min(30, max(5, len(features) - 1))  # Ensure valid perplexity
            print(f"[DEBUG] Using perplexity={perplexity} for t-SNE")
            
            # Use a dynamic seed based on checkpoint and client_id to ensure different visualizations
            import hashlib
            checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SSL_checkpoints', 'final_checkpoint.pt')
            checkpoint_modified_time = os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0
            random_seed = (int(checkpoint_modified_time * 1000) + (client_id or 0) * 100) % 10000
            print(f"[DEBUG] Using dynamic random_state={random_seed} for t-SNE based on checkpoint modification time")
            
            tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity)
            features_2d = tsne.fit_transform(features)
            
            # Create scatter plot
            plt.figure(figsize=(12, 10))
            
            # Define a color map
            colors = plt.cm.tab10.colors
            
            # Plot each class with a different color
            for class_idx in sorted(set(labels)):
                mask = labels == class_idx
                if np.any(mask):  # Only plot if we have samples for this class
                    plt.scatter(
                        features_2d[mask, 0],
                        features_2d[mask, 1],
                        c=[colors[int(class_idx) % len(colors)]],
                        s=50,
                        alpha=0.7,
                        label=f"Class {class_idx}"
                    )
            
            plt.title("SSL Encoder Embeddings of Labeled Data", fontsize=16)
            plt.xlabel("t-SNE Component 1", fontsize=14)
            plt.ylabel("t-SNE Component 2", fontsize=14)
            plt.legend(fontsize=12)
            
            # Save the plot
            plt.tight_layout()
            client_str = f"_client{client_id}" if client_id is not None else ""
            filename = os.path.join(output_dir, f"labeled_embeddings{client_str}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            
            print(f"[SSLEntropy] Visualization saved to {filename}")
            
            # Analyze cluster quality if we have multiple classes
            if len(set(labels)) > 1:
                self.analyze_cluster_quality(features_2d, labels, output_dir, client_id)
            
            print(f"[DEBUG] Visualization completed successfully")
                
        except ImportError as e:
            print(f"[SSLEntropy] Warning: Could not create visualization due to missing dependencies: {e}")
            print("[SSLEntropy] Please install matplotlib and scikit-learn to enable visualizations")
        except Exception as e:
            print(f"[ERROR] Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error stack
    
    def analyze_cluster_quality(self, features_2d, labels, output_dir='labeled_embeddings', client_id=None):
        """
        Analyze the quality of clusters in the t-SNE visualization
        
        Args:
            features_2d: 2D features from t-SNE
            labels: True labels
            output_dir: Directory to save results
            client_id: Client ID for naming files
        """
        print(f"[DEBUG] Analyzing cluster quality for {len(features_2d)} samples with {len(set(labels))} classes")
        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            import os
            
            metrics = {}
            
            # Silhouette score (higher is better, range: -1 to 1)
            metrics['silhouette'] = silhouette_score(features_2d, labels)
            
            # Davies-Bouldin index (lower is better)
            metrics['davies_bouldin'] = davies_bouldin_score(features_2d, labels)
            
            # Calinski-Harabasz index (higher is better)
            metrics['calinski_harabasz'] = calinski_harabasz_score(features_2d, labels)
            
            print("[SSLEntropy] Cluster quality metrics:")
            print(f"  Silhouette Score: {metrics['silhouette']:.4f} (higher is better)")
            print(f"  Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} (lower is better)")
            print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f} (higher is better)")
            
            # Save metrics to file
            client_str = f"_client{client_id}" if client_id is not None else ""
            filename = os.path.join(output_dir, f"cluster_metrics{client_str}.txt")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write("Cluster quality metrics:\n")
                f.write(f"Silhouette Score: {metrics['silhouette']:.4f} (higher is better)\n")
                f.write(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f} (lower is better)\n")
                f.write(f"Calinski-Harabasz Index: {metrics['calinski_harabasz']:.4f} (higher is better)\n")
            
            print(f"[SSLEntropy] Metrics saved to {filename}")
            print(f"[DEBUG] Cluster quality analysis completed successfully")
            
            return metrics
        
        except ImportError as e:
            print(f"[SSLEntropy] Warning: Could not analyze cluster quality due to missing dependencies: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Cluster quality analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error stack
            return None
            
    def analyze_client_labeled_data(self, model, dataset, client_indices, client_id=0, labeled_fraction=0.1, output_dir='labeled_embeddings'):
        """
        Analyze a client's labeled data by creating visualizations of embeddings and cluster quality
        
        Args:
            model: Current model
            dataset: Dataset containing the samples
            client_indices: List of indices that belong to this client
            client_id: ID of the client for naming files
            labeled_fraction: Fraction of client's data to use as labeled (default: 10%)
            output_dir: Directory to save visualizations
        """
        import numpy as np
        from torch.utils.data import DataLoader
        import os
        
        print(f"\n[SSLEntropy] Analyzing client {client_id}'s labeled data...")
        
        # Create output directory
        client_dir = os.path.join(output_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        
        # Sample labeled_fraction of the client's data
        labeled_size = int(labeled_fraction * len(client_indices))
        # Use a fixed seed for reproducibility
        np.random.seed(client_id + 42)
        labeled_indices = np.random.choice(client_indices, labeled_size, replace=False).tolist()
        
        print(f"[SSLEntropy] Client {client_id} has {len(client_indices)} total samples")
        print(f"[SSLEntropy] Using {len(labeled_indices)} samples ({labeled_fraction*100:.1f}%) as labeled data")
        
        # Create dataloader for labeled data
        labeled_loader = DataLoader(
            dataset,
            batch_size=64,
            sampler=SubsetSequentialSampler(labeled_indices),
            num_workers=2
        )
        
        # Visualize labeled embeddings
        self.visualize_labeled_embeddings(
            model, 
            labeled_loader, 
            output_dir=client_dir,
            client_id=client_id
        )
        
        # Calculate and save class distribution statistics
        try:
            # Get class labels
            client_labels = np.array([dataset[i][1] for i in client_indices])
            labeled_labels = np.array([dataset[i][1] for i in labeled_indices])
            num_classes = max(10, len(np.unique(client_labels)))
            
            # Calculate distributions
            full_dist = np.bincount(client_labels, minlength=num_classes) / len(client_labels)
            labeled_dist = np.bincount(labeled_labels, minlength=num_classes) / len(labeled_labels)
            
            # Save to file
            with open(os.path.join(client_dir, f"distribution_stats_client{client_id}.txt"), 'w') as f:
                f.write(f"Client {client_id} Data Distribution Statistics:\n\n")
                
                f.write("Full client data distribution:\n")
                for cls in range(num_classes):
                    if np.sum(client_labels == cls) > 0 or np.sum(labeled_labels == cls) > 0:
                        f.write(f"  Class {cls}: {np.sum(client_labels == cls)} samples ({full_dist[cls]*100:.1f}%)\n")
                    
                f.write("\nLabeled data distribution:\n")
                for cls in range(num_classes):
                    if np.sum(labeled_labels == cls) > 0:
                        f.write(f"  Class {cls}: {np.sum(labeled_labels == cls)} samples ({labeled_dist[cls]*100:.1f}%)\n")
                        
            print(f"[SSLEntropy] Distribution statistics saved to {os.path.join(client_dir, f'distribution_stats_client{client_id}.txt')}")
            
            # Print distribution summary
            print("[SSLEntropy] Classes represented in labeled data:")
            classes_with_samples = []
            for cls in range(num_classes):
                if np.sum(labeled_labels == cls) > 0:
                    count = np.sum(labeled_labels == cls)
                    percentage = count / len(labeled_labels) * 100
                    classes_with_samples.append(cls)
                    print(f"  Class {cls}: {count} samples ({percentage:.1f}%)")
                    
            print(f"[SSLEntropy] Client {client_id} has {len(classes_with_samples)}/{num_classes} classes in labeled data")
            
        except Exception as e:
            print(f"[SSLEntropy] Warning: Could not calculate distribution statistics: {e}")
        
        return labeled_indices
            
    def dirichlet_partition_data(self, dataset, num_clients=5, alpha=0.5, num_classes=10, seed=42):
        """
        Partition data using Dirichlet distribution to create non-IID data splits
        
        Args:
            dataset: Dataset to partition
            num_clients: Number of clients to create partitions for
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            num_classes: Number of classes in the dataset
            seed: Random seed for reproducibility
            
        Returns:
            Dict mapping client_id to list of indices for that client
        """
        import numpy as np
        
        np.random.seed(seed)
        
        # Get labels for entire dataset
        labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        # Initialize client indices dict
        client_indices = {i: [] for i in range(num_clients)}
        
        # For each class, distribute samples among clients according to Dirichlet distribution
        for cls in range(num_classes):
            # Get indices of samples from this class
            class_indices = np.where(labels == cls)[0]
            
            # Sample Dirichlet distribution for this class
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            # Calculate number of samples per client for this class
            # Ensure we distribute all samples
            num_samples_per_client = np.floor(proportions * len(class_indices)).astype(int)
            remainder = len(class_indices) - np.sum(num_samples_per_client)
            
            # Add remainder to clients with highest proportions
            if remainder > 0:
                sorted_proportions = np.argsort(proportions)[::-1]
                for i in range(remainder):
                    num_samples_per_client[sorted_proportions[i]] += 1
                    
            # Assign samples to clients
            class_indices_permuted = np.random.permutation(class_indices)
            start_idx = 0
            for client_id, num_samples in enumerate(num_samples_per_client):
                client_indices[client_id].extend(
                    class_indices_permuted[start_idx:start_idx + num_samples].tolist()
                )
                start_idx += num_samples
        
        # Shuffle indices for each client
        for client_id in client_indices.keys():
            np.random.shuffle(client_indices[client_id])
        
        # Print distribution statistics
        print("\n[SSLEntropy] Dirichlet partition statistics:")
        print(f"[SSLEntropy] Alpha: {alpha} (lower = more non-IID)")
        
        for client_id in range(num_clients):
            client_labels = labels[client_indices[client_id]]
            dist = np.bincount(client_labels, minlength=num_classes) / len(client_labels)
            print(f"[SSLEntropy] Client {client_id}: {len(client_indices[client_id])} samples")
            print(f"[SSLEntropy]   Class distribution: {dist}")
            
        return client_indices
        
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
        
    def analyze_federated_data(self, dataset, model, num_clients=5, alpha=0.5, client_ids=None, labeled_fraction=0.1, output_dir='ssl_embeddings'):
        """
        Comprehensive analysis of federated data with Dirichlet partitioning
        
        Args:
            dataset: Dataset to analyze
            model: Current model
            num_clients: Total number of clients
            alpha: Dirichlet concentration parameter
            client_ids: List of specific client IDs to analyze (default: all clients)
            labeled_fraction: Fraction of each client's data to use as labeled
            output_dir: Base directory for saving visualizations
            
        Returns:
            Dict mapping client_id to labeled indices
        """
        # Create directory for results
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Partition data using Dirichlet distribution
        client_indices = self.dirichlet_partition_data(
            dataset, 
            num_clients=num_clients, 
            alpha=alpha
        )
        
        # If no client_ids specified, analyze all clients
        if client_ids is None:
            client_ids = list(client_indices.keys())
        
        # Analyze each client's labeled data
        labeled_indices_by_client = {}
        for client_id in client_ids:
            if client_id not in client_indices:
                print(f"[SSLEntropy] Warning: Client {client_id} not found in partition, skipping")
                continue
                
            print(f"\n[SSLEntropy] Analyzing client {client_id}...")
            labeled_indices = self.analyze_client_labeled_data(
                model,
                dataset,
                client_indices[client_id],
                client_id=client_id,
                labeled_fraction=labeled_fraction,
                output_dir=output_dir
            )
            
            labeled_indices_by_client[client_id] = labeled_indices
            
        return labeled_indices_by_client