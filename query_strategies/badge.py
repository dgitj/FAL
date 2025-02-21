import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

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
        Computes gradient embeddings for the unlabeled data.
        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
        Returns:
            tuple: Gradient embeddings and their corresponding indices.
        """
        model.eval()
        gradient_embeddings = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(unlabeled_loader):
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                    if hasattr(batch[1], 'item'):  # If second element is indices
                        batch_indices = batch[1].tolist()
                    else:
                        batch_indices = list(range(batch_idx * unlabeled_loader.batch_size,
                                               min((batch_idx + 1) * unlabeled_loader.batch_size,
                                                   len(unlabeled_loader.dataset))))
                else:
                    inputs = batch.to(self.device)
                    batch_indices = list(range(batch_idx * unlabeled_loader.batch_size,
                                           min((batch_idx + 1) * unlabeled_loader.batch_size,
                                               len(unlabeled_loader.dataset))))
                
                # Forward pass to get predictions
                try:
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Take first element if model returns multiple outputs
                except:
                    raise ValueError("Model forward pass failed. Check your model architecture.")
                
                probabilities = F.softmax(outputs, dim=-1)
                pseudo_labels = probabilities.max(dim=1)[1]
                
                # Compute gradients for each sample
                for i in range(inputs.size(0)):
                    x = inputs[i:i+1].clone().requires_grad_(True)
                    
                    # Forward pass for single sample
                    output = model(x)
                    if isinstance(output, tuple):
                        output = output[0]
                        
                    loss = F.cross_entropy(output, pseudo_labels[i:i+1])
                    
                    # Compute gradients
                    model.zero_grad()
                    loss.backward()
                    
                    # Extract gradients from final layer (output layer)
                    grad_embedding = []
                    for name, param in model.named_parameters():
                        if 'weight' in name and param.requires_grad:
                            if param.grad is not None:
                                grad_embedding.append(param.grad.flatten().detach().cpu().numpy())
                    
                    if grad_embedding:
                        # Concatenate all gradients into one vector
                        grad_embedding = np.concatenate(grad_embedding)
                        gradient_embeddings.append(grad_embedding)
                        indices.append(batch_indices[i])
                    
        return np.array(gradient_embeddings), indices
        
    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples):
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
        
        if len(gradients) == 0:
            # Fallback to random selection if no gradients computed
            selected_indices = np.random.choice(len(unlabeled_set), num_samples, replace=False)
            selected_samples = [unlabeled_set[i] for i in selected_indices]
            remaining_unlabeled = [idx for i, idx in enumerate(unlabeled_set) if i not in selected_indices]
            return selected_samples, remaining_unlabeled
            
        # Perform k-means++ clustering to find diverse points
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
                distances[closest_idx] = np.inf
                next_closest = np.argmin(distances)
                selected_indices.append(next_closest)
        
        # Get the actual dataset indices corresponding to selected embeddings
        selected_samples = [data_indices[idx] for idx in selected_indices]
        
        # Update remaining unlabeled set
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        return selected_samples, remaining_unlabeled