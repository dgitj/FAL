import numpy as np
import torch
from sklearn.metrics import pairwise_distances

class CoreSetSampler:
    def __init__(self, device="cuda"):
        """
        Initializes the Core-Set active learning sampler.
        
        Core-Set selects samples to minimize the maximum distance between 
        any unlabeled point and a labeled point in the feature space.
        
        Args:
            device (str): Device to run calculations on (e.g., 'cuda' or 'cpu').
        """
        # Check if CUDA is available, fallback to CPU if not
        if device == "cuda" and not torch.cuda.is_available():
            print("[CoreSet] CUDA not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
            
        self.debug = True  # Enable detailed debugging
        print(f"[CoreSet] Using device: {self.device}")

    def _extract_features(self, model, loader):
        """
        Extract features for all samples in the loader using the model.
        
        Args:
            model: PyTorch model to extract features from.
            loader: DataLoader containing samples.
            
        Returns:
            numpy.ndarray: Features for all samples.
        """
        model.eval()
        features = []
        indices = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                
                outputs = model(inputs)
                
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    batch_features = outputs[1]
                    
                    if isinstance(batch_features, list) and len(batch_features) > 0:
                        batch_features = batch_features[-1]  
                    
                else:
                    batch_features = outputs
                
                features.append(batch_features.cpu().numpy())
                
        return np.concatenate(features)
    
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
        selected = []
        
        # Calculate minimum distances for each unlabeled point to any labeled point
        if len(labeled_features) > 0:
            min_distances = pairwise_distances(unlabeled_features, labeled_features, metric='euclidean').min(axis=1)
        else:
            min_distances = pairwise_distances(unlabeled_features, unlabeled_features[0:1], metric='euclidean').min(axis=1)
        
        for _ in range(budget):
            idx = np.argmax(min_distances)
            
            selected.append(unlabeled_indices[idx])
            
            new_distances = pairwise_distances(unlabeled_features, unlabeled_features[idx:idx+1], metric='euclidean').reshape(-1)
            min_distances = np.minimum(min_distances, new_distances)
        
        return selected
    
    def select_samples(self, model, model_server=None, unlabeled_loader=None, client_id=None, unlabeled_set=None, num_samples=None, labeled_set=None, seed=None):
        """
        Selects samples using the Core-Set approach.
        
        Args:
            model (torch.nn.Module): Client model.
            model_server (torch.nn.Module, optional): Server model (not used in CoreSet).
            unlabeled_loader (DataLoader): Loader for unlabeled data.
            client_id (int, optional): ID of the client (not used in CoreSet).
            unlabeled_set (list): List of indices of unlabeled samples.
            num_samples (int): Number of samples to select.
            labeled_set (list, optional): List of indices of labeled samples.
            seed (int, optional): Random seed for reproducibility.
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        if self.debug:
            print(f"\n[CoreSet] Selecting {num_samples} samples")
            print(f"[CoreSet] Unlabeled pool size: {len(unlabeled_set)}")
        
        dataset = unlabeled_loader.dataset

        unlabeled_features = self._extract_features(model, unlabeled_loader)
        
        labeled_features = np.array([])
        
        if labeled_set is not None and len(labeled_set) > 0:
            print(f"[CoreSet] Using provided labeled set with {len(labeled_set)} samples")
            
            from torch.utils.data import DataLoader, Subset
            labeled_subset = Subset(dataset, labeled_set)
            labeled_loader = DataLoader(
                labeled_subset, 
                batch_size=unlabeled_loader.batch_size, 
                shuffle=False,
                num_workers=unlabeled_loader.num_workers if hasattr(unlabeled_loader, 'num_workers') else 0
            )
            
            # Extract features for labeled data
            labeled_features = self._extract_features(model, labeled_loader)
            
            if self.debug:
                print(f"[CoreSet] Extracted labeled features shape: {labeled_features.shape}")
        else:
            print(f"[CoreSet] No labeled set provided, starting from scratch.")
        
        if self.debug:
            print(f"[CoreSet] Extracted unlabeled features shape: {unlabeled_features.shape}")
        
        # Select samples using greedy k-center
        selected_samples = self._greedy_k_center(labeled_features, unlabeled_features, unlabeled_set, num_samples)
        
        if self.debug:
            print(f"[CoreSet] Selected {len(selected_samples)} samples")
        
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        return selected_samples, remaining_unlabeled
