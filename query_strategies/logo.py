import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


class LoGoSampler:
    def __init__(self, device="cuda"):
        """
        Initializes the LoGo sampler.
        Args:
            device (str): Device to run the calculations on.
        """
        self.device = device

    def extract_embeddings(self, model, unlabeled_loader):
        """
        Extracts feature embeddings from the local model with robust index tracking.

        Args:
            model (torch.nn.Module): Local model.
            unlabeled_loader (DataLoader): DataLoader for unlabeled data.

        Returns:
            tuple: (embeddings, original_indices) - Feature embeddings and their corresponding dataset indices.
        """
        model.eval()
        embeddings = []
        original_indices = []

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
                # Track original indices
                if hasattr(unlabeled_loader.sampler, 'indices'):
                    start_idx = batch_idx * unlabeled_loader.batch_size
                    end_idx = min((batch_idx + 1) * unlabeled_loader.batch_size, 
                                  len(unlabeled_loader.sampler.indices))
                    batch_indices = [unlabeled_loader.sampler.indices[i] for i in range(start_idx, end_idx)]
                else:
                    batch_indices = list(range(
                        batch_idx * unlabeled_loader.batch_size,
                        min((batch_idx + 1) * unlabeled_loader.batch_size, len(unlabeled_loader.dataset))
                    ))
                
                # Extract features
                inputs = inputs.to(self.device)
                try:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        _, features = outputs
                        if isinstance(features, list):
                            features = features[-1]  
                    else:
                        raise ValueError("Model output format not compatible with LoGo")
                    
                    # Store embeddings and corresponding indices
                    embeddings.append(features.cpu())
                    original_indices.extend(batch_indices)
                except Exception as e:
                    print(f"Error extracting embeddings: {str(e)}")
                    continue

        if len(embeddings) == 0:
            error_msg = "No valid embeddings extracted"
            print(error_msg)
            raise ValueError(error_msg)

        try:
            all_embeddings = torch.cat(embeddings, dim=0)
            if len(all_embeddings) != len(original_indices):
                print(f"Warning: Embedding count ({len(all_embeddings)}) doesn't match index count ({len(original_indices)})")
                min_len = min(len(all_embeddings), len(original_indices))
                all_embeddings = all_embeddings[:min_len]
                original_indices = original_indices[:min_len]
            
            return all_embeddings, original_indices
        except Exception as e:
            print(f"Error concatenating embeddings: {str(e)}")
            error_msg = f"Error concatenating embeddings: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

    def macro_micro_clustering(self, model_server, unlabeled_loader, embeddings, original_indices, num_samples, seed=None):
        """
        Performs LoGo's macro (clustering) and micro (uncertainty-based selection) steps.

        Args:
            model_server (torch.nn.Module): Global model.
            unlabeled_loader (DataLoader): Full unlabeled DataLoader.
            embeddings (torch.Tensor): Feature embeddings.
            original_indices (list): Original dataset indices corresponding to embeddings.
            num_samples (int): Number of samples to select.

        Returns:
            list: Selected sample indices.
        """
        if len(embeddings) == 0 or not original_indices:
            error_msg = "No embeddings or indices available for clustering in LoGo algorithm"
            print(error_msg)
            raise ValueError(error_msg)
            
        # Ensure we don't create more clusters than samples
        num_clusters = min(num_samples, len(embeddings))
        
        try:
            # Macro step: K-Means clustering
            kmeans_seed = seed if seed is not None else 0
            kmeans = KMeans(n_clusters=num_clusters, random_state=kmeans_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings.numpy())
            
            # Group samples by cluster
            clusters = [[] for _ in range(num_clusters)]
            for i, cluster_id in enumerate(cluster_labels):
                clusters[cluster_id].append(original_indices[i])
                
            # Micro step: Calculate uncertainty for each cluster
            selected_samples = []
            model_server.eval()
            
            for cluster_indices in clusters:
                if not cluster_indices:
                    continue
                worker_init_fn = None
                if seed is not None:
                    def worker_seed_fn(worker_id):
                        worker_seed = seed + worker_id
                        np.random.seed(worker_seed)
                        torch.manual_seed(worker_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(worker_seed)
                    worker_init_fn = worker_seed_fn
                    
                generator = None
                if seed is not None:
                    generator = torch.Generator()
                    generator.manual_seed(seed)
                    
                # Create loader for this cluster with deterministic settings
                cluster_subset = torch.utils.data.Subset(unlabeled_loader.dataset, cluster_indices)
                cluster_loader = torch.utils.data.DataLoader(
                    cluster_subset, 
                    batch_size=min(32, len(cluster_indices)), 
                    shuffle=False, 
                    num_workers=0,  
                    worker_init_fn=worker_init_fn,
                    generator=generator
                )
                

                # Calculate uncertainty for each sample in the cluster
                max_uncertainty = -float('inf')
                most_uncertain_sample = None
                sample_idx = 0
                
                with torch.no_grad():
                    for inputs, _ in cluster_loader:
                        inputs = inputs.to(self.device)
                        
                        # Get model outputs
                        outputs = model_server(inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        
                        # Calculate entropy
                        probs = F.softmax(outputs, dim=1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
                        
                        # Find max entropy in this batch
                        for i, uncertainty in enumerate(entropy):
                            uncertainty_val = uncertainty.item()
                            if uncertainty_val > max_uncertainty:
                                max_uncertainty = uncertainty_val
                                most_uncertain_sample = cluster_indices[sample_idx + i]
                        
                        sample_idx += inputs.size(0)
                
                if most_uncertain_sample is not None:
                    selected_samples.append(most_uncertain_sample)
            
            return selected_samples
            
        except Exception as e:
            error_msg = f"Error in LoGo clustering: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

    def select_samples(self, model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, seed=None):
        """
        Selects samples using the LoGo strategy with robust index tracking.

        Args:
            model (torch.nn.Module): Local model.
            model_server (torch.nn.Module): Global model.
            unlabeled_loader (DataLoader): DataLoader for unlabeled data.
            c (int): Client ID (unused, kept for compatibility).
            unlabeled_set (list): Indices of unlabeled data.
            num_samples (int): Number of samples to select.

        Returns:
            tuple: Selected samples and remaining unlabeled samples.
        """
        num_samples = min(num_samples, len(unlabeled_set))
        
        try:
            embeddings, original_indices = self.extract_embeddings(model, unlabeled_loader)
            
            if len(embeddings) == 0 or not original_indices:
                error_msg = "Failed to extract embeddings for LoGo algorithm"
                print(error_msg)
                raise ValueError(error_msg)
                
            valid_indices = []
            valid_embeddings = []
            
            for i, idx in enumerate(original_indices):
                if idx in unlabeled_set:
                    valid_indices.append(idx)
                    valid_embeddings.append(embeddings[i])
            
            if not valid_indices:
                error_msg = "No valid indices found in unlabeled_set for LoGo algorithm"
                print(error_msg)
                raise ValueError(error_msg)
                
            if valid_embeddings:
                valid_embeddings = torch.stack(valid_embeddings)
                
            # Perform clustering and uncertainty-based selection
                selected_samples = self.macro_micro_clustering(
                model_server, 
                unlabeled_loader, 
                valid_embeddings, 
                valid_indices, 
                num_samples,
                seed
            )
            
            if len(selected_samples) < num_samples:
                raise ValueError(f"LoGo algorithm could not select enough samples. Requested {num_samples}, but only found {len(selected_samples)}.")
            
            # Get remaining unlabeled samples
            remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
            
            return selected_samples, remaining_unlabeled
            
        except Exception as e:
            error_msg = f"Error in LoGo sample selection: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e