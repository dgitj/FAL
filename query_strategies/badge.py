import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from torch.cuda.amp import autocast

class BADGESampler:
    def __init__(self, device="cuda"):
        """
        Initializes the BADGE sampler.
        Args:
            device (str): Device to run the calculations on (e.g., 'cuda' or 'cpu').
        """
        self.device = device

    def compute_gradient_embeddings(self, model, unlabeled_loader):
        """
        Compute gradient embeddings for the BADGE sampling strategy.
        
        Args:
            model: The neural network model
            unlabeled_loader: DataLoader containing unlabeled samples
            
        Returns:
            tuple: (gradient_embeddings, data_indices)
        """
        

        model.eval()  # Set model to evaluation mode
        gradients = []
        data_indices = []

        with autocast(enabled=True):
            for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
                inputs = inputs.to(self.device)
                inputs.requires_grad_(True) # track all operations on these inputs

                # Forward pass with proper output handling
                model_output = model(inputs)
                
                # Extract logits based on model output format
                if isinstance(model_output, tuple):
                    # Handle case where model returns (logits, features) or (logits, [features])
                    outputs = model_output[0]
                else:
                    # Handle case where model directly returns logits
                    outputs = model_output

                # Create virtual labels from predictions (single operation)
                probs = F.softmax(outputs, dim=1)
                grad_embedding = torch.zeros_like(probs)
                virtual_labels = probs.max(1)[1] # For each sample find the index of the class with the highest probability
                grad_embedding.scatter_(1, virtual_labels.unsqueeze(1), 1)
                
                # Single backward pass for all classes
                loss = -(grad_embedding * outputs).sum()
                loss.backward()
                
                # Store gradients and indices
                grad = inputs.grad.view(inputs.size(0), -1)
                gradients.append(grad.cpu().detach())
                
                # Calculate correct indices in the dataset
                batch_indices = list(range(
                    batch_idx * unlabeled_loader.batch_size,
                    min((batch_idx + 1) * unlabeled_loader.batch_size, len(unlabeled_loader.dataset))
                ))
                data_indices.extend(batch_indices)
                
                # Clean up gradients
                inputs.grad = None

        # Ensure we have gradients to return
        if len(gradients) == 0:
            print("Warning: No gradients computed in BADGE sampling")
            return torch.zeros(0, 1), []
            
        # Concatenate all batch gradients
        try:
            gradients = torch.cat(gradients, dim=0)
        except RuntimeError as e:
            print(f"Error concatenating gradients: {e}")
            return torch.zeros(0, 1), []
            
        return gradients, data_indices


    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples, seed=None):
        """
        Selects samples using BADGE sampling strategy.
        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            unlabeled_set (list): List of indices corresponding to the unlabeled data.
            num_samples (int): Number of samples to select.
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        # Ensure we don't request more samples than available
        num_samples = min(num_samples, len(unlabeled_set))
        
        # Get embeddings and their corresponding dataset indices
        gradients, data_indices = self.compute_gradient_embeddings(model, unlabeled_loader)
        
        if len(gradients) == 0 or len(data_indices) == 0:
            raise ValueError("BADGE computation failed: No badge scores were generated")
            
        # Handle case where indices don't match
        if len(gradients) != len(unlabeled_set):
            raise ValueError(f"BADGE computation error: Gradient count ({len(gradients)}) doesn't match unlabeled set size ({len(unlabeled_set)}). This may be due to batch size issues or dataloader configuration.")
        
        # Perform k-means++ clustering to find diverse points
        try:
            kmeans = KMeans(n_clusters=num_samples, init="k-means++", n_init=10, random_state=42)
            kmeans.fit(gradients)
            
            # Find points closest to each cluster center
            centers = kmeans.cluster_centers_
            selected_indices = []
            
            for center in centers:
                distances = np.linalg.norm(gradients - center, axis=1)
                closest_idx = np.argmin(distances)
                
                # Avoid selecting the same point twice
                if closest_idx not in selected_indices:
                    selected_indices.append(closest_idx)
                else:
                    # Find the next closest point
                    distances[closest_idx] = np.inf # avoid selecting the same sample for several clusters
                    next_closest = np.argmin(distances)
                    selected_indices.append(next_closest)
            
            # Get the actual dataset indices corresponding to selected embeddings
            selected_samples = [unlabeled_set[idx] for idx in selected_indices]
            
            # Update remaining unlabeled set
            remaining_unlabeled = [idx for i, idx in enumerate(unlabeled_set) if i not in selected_indices]
            
            return selected_samples, remaining_unlabeled
            
        except Exception as e:
            raise ValueError(f"BADGE sampling failed: Error in K-means clustering: {e}. Try reducing the number of samples to select or check your data.")