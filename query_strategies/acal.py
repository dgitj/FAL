import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
import config
import numpy as np
import random
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


class ACALSampler:
    def __init__(self, device="cuda", js_threshold=0.08, js_sample_size=2000):
        """
        ACAL: Adaptive Curriculum Active Learning
        
        Implements the ACAL strategy from MICCAI 2024 paper:
        "Adaptive Curriculum Query Strategy for Active Learning in Medical Image Classification"
        
        Uses curriculum learning principles:
        - Phase 1 (Diversity): Random sampling to cover various difficulty levels
        - Phase 2 (Uncertainty): Entropy-based sampling for hard examples
        
        Switches from Phase 1 to Phase 2 when JS divergence < threshold
        
        Args:
            device (str): Device to run calculations on
            js_threshold (float): JS divergence threshold for phase switching (default: 0.08)
            js_sample_size (int): Number of samples to use for JS divergence calculation (default: 2000)
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("[ACAL] CUDA not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        
        self.js_threshold = js_threshold
        self.js_sample_size = js_sample_size
        self.phase = "diversity"  # Start with diversity phase
        self.debug = False
        
        print(f"[ACAL] Initialized with JS threshold: {js_threshold}")
        print(f"[ACAL] JS divergence sample size: {js_sample_size}")
        print(f"[ACAL] Phase 1: Random sampling (diversity)")
        print(f"[ACAL] Phase 2: Entropy sampling (uncertainty)")
    
    def compute_js_divergence(self, all_embeddings, labeled_embeddings, bins=50):
        """
        Compute Jensen-Shannon divergence between distributions using per-feature approach.
        
        Measures similarity between labeled set distribution and full dataset distribution
        by computing JS divergence for each feature dimension separately and averaging.
        This matches the original ACAL paper implementation.
        
        Lower values indicate more similar distributions.
        
        Args:
            all_embeddings (np.ndarray): Embeddings of all training samples, shape (N, D)
            labeled_embeddings (np.ndarray): Embeddings of labeled samples, shape (M, D)
            bins (int): Number of bins for histogram computation (default: 50)
            
        Returns:
            float: Average JS divergence across all feature dimensions
        """
        # Get number of features
        num_features = all_embeddings.shape[1]
        js_divergences = np.zeros(num_features)
        
        # Compute JS divergence for each feature dimension
        for i in range(num_features):
            # Get feature values for current dimension
            all_feature = all_embeddings[:, i]
            labeled_feature = labeled_embeddings[:, i]
            
            # Create bins for histogram based on min/max of both distributions
            min_val = min(all_feature.min(), labeled_feature.min())
            max_val = max(all_feature.max(), labeled_feature.max())
            
            # Handle edge case where min == max
            if min_val == max_val:
                js_divergences[i] = 0.0
                continue
            
            bins_range = np.linspace(min_val, max_val, bins)
            
            # Compute histograms (probability distributions)
            all_hist, _ = np.histogram(all_feature, bins=bins_range, density=True)
            labeled_hist, _ = np.histogram(labeled_feature, bins=bins_range, density=True)
            
            # Normalize to ensure they sum to 1
            all_hist = all_hist / (all_hist.sum() + 1e-10)
            labeled_hist = labeled_hist / (labeled_hist.sum() + 1e-10)
            
            # Add small epsilon to avoid log(0)
            all_hist = all_hist + 1e-10
            labeled_hist = labeled_hist + 1e-10
            
            # Compute JS divergence for this feature
            js_divergences[i] = jensenshannon(all_hist, labeled_hist)
        
        # Return mean JS divergence across all features
        mean_js_div = float(np.mean(js_divergences))
        
        if self.debug:
            print(f"[ACAL] JS divergence per feature - min: {js_divergences.min():.6f}, "
                  f"max: {js_divergences.max():.6f}, mean: {mean_js_div:.6f}")
        
        return mean_js_div
    
    def get_embeddings(self, model, data_loader):
        """
        Extract embeddings from model for given data.
        
        Args:
            model: The neural network model
            data_loader: DataLoader containing the data
            
        Returns:
            np.ndarray: Extracted embeddings
        """
        model.eval()
        embeddings_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass - handle models that return tuple (outputs, embeddings)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    _, embeddings = outputs
                else:
                    # If model doesn't return embeddings separately, use outputs as embeddings
                    embeddings = outputs
                
                # Handle different embedding types
                if isinstance(embeddings, list):
                    # If embeddings is a list of tensors, concatenate them
                    embeddings = torch.cat([e.flatten(start_dim=1) if e.dim() > 2 else e for e in embeddings], dim=1)
                elif torch.is_tensor(embeddings) and embeddings.dim() > 2:
                    # If embeddings has more than 2 dimensions, flatten to (batch_size, features)
                    embeddings = embeddings.flatten(start_dim=1)
                
                embeddings_list.append(embeddings.cpu().numpy())
        
        return np.concatenate(embeddings_list, axis=0)
    
    def random_sampling(self, unlabeled_set, num_samples, seed=None):
        """
        Phase 1: Random sampling for diversity.
        
        Selects samples randomly to ensure coverage of various difficulty levels
        and broad representation of the data distribution.
        
        Args:
            unlabeled_set (list): Indices of unlabeled samples
            num_samples (int): Number of samples to select
            seed (int): Random seed for reproducibility
            
        Returns:
            list: Selected sample indices
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Random selection without replacement
        selected = np.random.choice(
            unlabeled_set, 
            size=min(num_samples, len(unlabeled_set)), 
            replace=False
        )
        
        return selected.tolist()
    
    def entropy_sampling(self, model, unlabeled_loader, unlabeled_set, num_samples, seed=None):
        """
        Phase 2: Entropy-based sampling for uncertainty.
        
        Selects samples with highest prediction entropy (most uncertain).
        Focuses on hard-to-classify examples after model has learned basic patterns.
        
        Args:
            model: The neural network model
            unlabeled_loader: DataLoader for unlabeled data
            unlabeled_set (list): Indices of unlabeled samples
            num_samples (int): Number of samples to select
            seed (int): Random seed for reproducibility
            
        Returns:
            list: Selected sample indices
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        model.eval()
        uncertainties = []
        indices = []
        
        with torch.no_grad():
            batch_idx = 0
            for batch in unlabeled_loader:
                batch_idx += 1
                
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Get batch indices
                batch_indices = unlabeled_loader.sampler.indices[
                    (batch_idx - 1) * unlabeled_loader.batch_size:
                    min(batch_idx * unlabeled_loader.batch_size, len(unlabeled_loader.sampler))
                ]
                indices.extend(batch_indices)
                
                # Forward pass - handle models that return tuple (outputs, embeddings)
                model_output = model(inputs)
                if isinstance(model_output, tuple):
                    outputs = model_output[0]
                else:
                    outputs = model_output
                
                # Calculate probabilities
                probs = F.softmax(outputs, dim=1)
                
                # Calculate entropy: -sum(p * log(p))
                log_probs = F.log_softmax(outputs, dim=1)
                entropy_values = -torch.sum(probs * log_probs, dim=1)
                
                uncertainties.extend(entropy_values.cpu().numpy())
        
        indices = np.array(indices)
        uncertainties = np.array(uncertainties)
        
        # Select top-k samples with highest entropy (most uncertain)
        top_indices = np.argsort(uncertainties)[-num_samples:]
        selected = indices[top_indices]
        
        return selected.tolist()
    
    def select_samples(self, model, model_server, unlabeled_loader, client_id, 
                      unlabeled_set, num_samples, labeled_set=None, seed=None, **kwargs):
        """
        Main sample selection method implementing ACAL algorithm.
        
        Monitors JS divergence and adaptively switches from diversity to uncertainty sampling.
        
        Args:
            model: Client model
            model_server: Server model (not used in ACAL, but kept for interface compatibility)
            unlabeled_loader: DataLoader for unlabeled data
            client_id (int): Client identifier
            unlabeled_set (list): Indices of unlabeled samples
            num_samples (int): Number of samples to select
            labeled_set (list): Indices of labeled samples
            seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        # Set random seeds for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        if self.debug:
            print(f"\n[ACAL] Client {client_id}: Selecting {num_samples} samples")
            print(f"[ACAL] Current phase: {self.phase}")
            print(f"[ACAL] Unlabeled pool size: {len(unlabeled_set)}")
        
        # Step 1: Extract embeddings - use subset for efficiency
        dataset = unlabeled_loader.dataset
        
        # Sample a subset for JS divergence calculation instead of using all data
        dataset_size = len(dataset)
        if dataset_size > self.js_sample_size:
            # Randomly sample indices for JS calculation
            sample_indices = np.random.choice(dataset_size, self.js_sample_size, replace=False).tolist()
            if self.debug:
                print(f"[ACAL] Using {self.js_sample_size} samples (out of {dataset_size}) for JS divergence")
        else:
            sample_indices = list(range(dataset_size))
            if self.debug:
                print(f"[ACAL] Using all {dataset_size} samples for JS divergence")
        
        # Create loader for sampled data
        sample_loader = DataLoader(
            dataset,
            batch_size=config.BATCH,
            sampler=SubsetSequentialSampler(sample_indices),
            num_workers=0,
            pin_memory=True
        )
        
        # Extract embeddings from sample
        sample_embeddings = self.get_embeddings(model, sample_loader)
        
        if labeled_set is not None and len(labeled_set) > 0:
            # Create loader for labeled data
            labeled_loader = DataLoader(
                dataset,
                batch_size=config.BATCH,
                sampler=SubsetSequentialSampler(labeled_set),
                num_workers=0,
                pin_memory=True
            )
            labeled_embeddings = self.get_embeddings(model, labeled_loader)
            
            # Step 2: Compute JS divergence using sample
            js_divergence = self.compute_js_divergence(sample_embeddings, labeled_embeddings)
            
            if self.debug:
                print(f"[ACAL] JS divergence: {js_divergence:.6f} (threshold: {self.js_threshold})")
            
            # Step 3: Check if we should switch phases
            if self.phase == "diversity" and js_divergence < self.js_threshold:
                self.phase = "uncertainty"
                print(f"\n[ACAL] *** PHASE SWITCH *** Client {client_id}")
                print(f"[ACAL] JS divergence {js_divergence:.6f} < threshold {self.js_threshold}")
                print(f"[ACAL] Switching from DIVERSITY (Random) to UNCERTAINTY (Entropy)")
                print(f"[ACAL] Labeled set size: {len(labeled_set)}")
        else:
            # First round, no labeled data yet - stay in diversity phase
            js_divergence = float('inf')
            if self.debug:
                print(f"[ACAL] First round, no labeled data - using diversity sampling")
        
        # Step 4: Sample based on current phase
        if self.phase == "diversity":
            if self.debug:
                print(f"[ACAL] Using RANDOM sampling (Phase 1: Diversity)")
            selected_samples = self.random_sampling(unlabeled_set, num_samples, seed)
        else:  # uncertainty phase
            if self.debug:
                print(f"[ACAL] Using ENTROPY sampling (Phase 2: Uncertainty)")
            selected_samples = self.entropy_sampling(
                model, unlabeled_loader, unlabeled_set, num_samples, seed
            )
        
        # Calculate remaining unlabeled samples
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        if self.debug:
            print(f"[ACAL] Selected {len(selected_samples)} samples")
            print(f"[ACAL] Remaining unlabeled: {len(remaining_unlabeled)}")
        
        return selected_samples, remaining_unlabeled