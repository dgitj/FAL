import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import DataLoader

class NoiseStabilitySampler:
    def __init__(self, device="cuda", noise_scale=0.001, num_sampling=50):
        """
        Initializes the Noise Stability Sampler.

        Args:
            device (str): Device to run the calculations on (e.g., 'cuda' or 'cpu').
            noise_scale (float): Scaling factor for noise perturbation. Default 0.001 from original paper
            num_sampling (int): Number of times noise is added to the model. Default 50 from original paper
        """
        self.device = device
        self.noise_scale = noise_scale
        self.num_sampling = num_sampling

    def add_noise_to_weights(self, model):
        """
        Adds Gaussian noise to model weights.
        
        Args:
            model: PyTorch model to perturb
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    # Calculate normalization factor to keep relative noise scale consistent
                    param_norm = param.norm()
                    if param_norm > 0:  # Avoid division by zero
                        noise = torch.randn_like(param) * self.noise_scale * param_norm / torch.norm(torch.randn_like(param))
                        param.add_(noise)

    def compute_uncertainty(self, model, unlabeled_loader):
        """
        Computes feature deviations before and after adding noise, with proper index tracking.

        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.

        Returns:
            tuple: (uncertainty_scores, original_indices) - Scores and their corresponding dataset indices
        """
        model.eval()
        
        try:
            # Get original outputs, features and indices
            outputs, features, original_indices = self.get_all_outputs(model, unlabeled_loader)
            if features is None or len(features) == 0:
                # Fallback to random uncertainty if feature extraction fails
                print("Warning: Failed to extract features in Noise Stability. Using random scores.")
                
                # Even with failure, try to get original indices if possible
                if not original_indices and hasattr(unlabeled_loader.sampler, 'indices'):
                    original_indices = unlabeled_loader.sampler.indices.copy()
                
                # Generate random scores with original indices if available
                if original_indices:
                    return torch.rand(len(original_indices)).to(self.device), original_indices
                else:
                    # Complete fallback with sequential indices
                    random_scores = torch.rand(len(unlabeled_loader.dataset)).to(self.device)
                    sequential_indices = list(range(len(unlabeled_loader.dataset)))
                    return random_scores, sequential_indices
            
            # Initialize difference tensor
            diffs = torch.zeros_like(features).to(self.device)

            # Apply noise multiple times and measure deviation
            successful_iterations = 0
            for i in range(self.num_sampling):
                # Create a deep copy of the model to avoid modifying the original
                noisy_model = copy.deepcopy(model).to(self.device)
                self.add_noise_to_weights(noisy_model)
                noisy_model.eval()
                
                # Get outputs from noisy model
                _, noisy_features, _ = self.get_all_outputs(noisy_model, unlabeled_loader)
                if noisy_features is None or noisy_features.shape != features.shape:
                    continue
                    
                # Calculate absolute difference
                diff_k = noisy_features - features
                diffs += diff_k.abs()
                successful_iterations += 1

            # Normalize by number of successful noise iterations
            if successful_iterations > 0:
                diffs = diffs / successful_iterations
                
            # Return mean difference across feature dimensions and original indices
            uncertainty = diffs.mean(dim=1)
            return uncertainty, original_indices
        
        except Exception as e:
            print(f"Error in compute_uncertainty: {str(e)}")
            # Fallback to random uncertainty
            if hasattr(unlabeled_loader.sampler, 'indices'):
                original_indices = unlabeled_loader.sampler.indices.copy()
                return torch.rand(len(original_indices)).to(self.device), original_indices
            else:
                random_scores = torch.rand(len(unlabeled_loader.dataset)).to(self.device)
                sequential_indices = list(range(len(unlabeled_loader.dataset)))
                return random_scores, sequential_indices

    def get_all_outputs(self, model, dataloader):
        """
        Runs the model on all samples and returns outputs, feature embeddings, and original indices.

        Args:
            model (torch.nn.Module): The model used for predictions.
            dataloader (DataLoader): The dataset loader.

        Returns:
            tuple: (outputs, features, original_indices) - Model outputs, feature embeddings and dataset indices
        """
        model.eval()
        outputs_list = []
        features_list = []
        original_indices = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)

                # Track original indices
                if hasattr(dataloader.sampler, 'indices'):
                    # Using SubsetSequentialSampler or similar
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = min((batch_idx + 1) * dataloader.batch_size, len(dataloader.sampler.indices))
                    batch_indices = [dataloader.sampler.indices[i] for i in range(start_idx, end_idx)]
                    original_indices.extend(batch_indices)
                else:
                    # Fallback if sampler doesn't have .indices attribute
                    batch_indices = list(range(
                        batch_idx * dataloader.batch_size,
                        min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                    ))
                    original_indices.extend(batch_indices)

                try:
                    # Forward pass through model
                    result = model(inputs)
                    
                    # Handle different model output formats
                    if isinstance(result, tuple) and len(result) >= 2:
                        logits, features = result[0], result[1]
                        
                        # If features is a list (e.g., block outputs), take the last one
                        if isinstance(features, list) and len(features) > 0:
                            features = features[-1]
                            
                        outputs_list.append(F.softmax(logits, dim=1))
                        features_list.append(features)
                    else:
                        # Model just returns logits, no features available
                        outputs_list.append(F.softmax(result, dim=1))
                        return torch.cat(outputs_list, dim=0), None, original_indices
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

        if not outputs_list or not features_list:
            return None, None, original_indices
            
        try:
            outputs = torch.cat(outputs_list, dim=0)
            features = torch.cat(features_list, dim=0)
            
            # Verify dimensions match
            if len(outputs) != len(original_indices) or len(features) != len(original_indices):
                print(f"Warning: Dimension mismatch in Noise Stability. "
                      f"Got {len(outputs)} outputs, {len(features)} features, but {len(original_indices)} indices")
                
            return outputs, features, original_indices
        except Exception as e:
            print(f"Error concatenating results: {str(e)}")
            return None, None, original_indices

    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples):
        """
        Selects the most uncertain samples based on feature deviation with proper index mapping.

        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            unlabeled_set (list): List of indices corresponding to the unlabeled data.
            num_samples (int): Number of samples to select.

        Returns:
            tuple: (selected_samples, remaining_unlabeled) - Lists of dataset indices
        """
        # Ensure we don't request more samples than available
        num_samples = min(num_samples, len(unlabeled_set))
        
        try:
            # Compute uncertainty scores with proper index tracking
            uncertainty, original_indices = self.compute_uncertainty(model, unlabeled_loader)
            
            # Validate the indices
            if uncertainty is None or len(uncertainty) == 0:
                # Fallback to random selection
                print("Warning: Uncertainty computation failed. Using random selection.")
                selected_indices = np.random.choice(len(unlabeled_set), num_samples, replace=False)
                selected_samples = [unlabeled_set[i] for i in selected_indices]
                remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
                return selected_samples, remaining_unlabeled
                
            # Check if computed indices match unlabeled_set
            if len(uncertainty) != len(original_indices):
                print(f"Warning: Mismatch between uncertainty scores ({len(uncertainty)}) and tracked indices ({len(original_indices)})")
                # Fallback to random selection
                selected_indices = np.random.choice(len(unlabeled_set), num_samples, replace=False)
                selected_samples = [unlabeled_set[i] for i in selected_indices]
                remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
                return selected_samples, remaining_unlabeled
                
            # Verify all original_indices are in unlabeled_set
            original_indices_set = set(original_indices)
            unlabeled_set_set = set(unlabeled_set)
            
            if not original_indices_set.issubset(unlabeled_set_set):
                print("Warning: Some tracked indices are not in unlabeled_set. Fixing...")
                # Filter to keep only indices that are in unlabeled_set
                valid_mask = torch.tensor([idx in unlabeled_set_set for idx in original_indices], 
                                         dtype=torch.bool)
                
                if valid_mask.sum() == 0:
                    print("Error: No valid indices found. Using random selection.")
                    selected_indices = np.random.choice(len(unlabeled_set), num_samples, replace=False)
                    selected_samples = [unlabeled_set[i] for i in selected_indices]
                    remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
                    return selected_samples, remaining_unlabeled
                
                # Filter uncertainty and indices
                filtered_uncertainty = uncertainty[valid_mask]
                filtered_indices = [idx for i, idx in enumerate(original_indices) if valid_mask[i]]
                
                # Use filtered values
                uncertainty = filtered_uncertainty
                original_indices = filtered_indices
            
            # Select indices with highest uncertainty
            uncertainty = uncertainty.cpu().numpy()
            sorted_idx = np.argsort(-uncertainty)  # Descending order
            
            # Get the actual dataset indices using the mapping
            selected_samples = [original_indices[i] for i in sorted_idx[:num_samples]]
            
            # Find the remaining unlabeled samples
            remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
            
            # Sanity check
            if len(selected_samples) != num_samples:
                print(f"Warning: Selected {len(selected_samples)} samples instead of {num_samples}")
            
            if len(set(selected_samples).intersection(set(remaining_unlabeled))) > 0:
                print("Warning: Overlap between selected and remaining samples")
                # Fix by deduplicating
                selected_samples = list(set(selected_samples))
                remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
            
            return selected_samples, remaining_unlabeled
            
        except Exception as e:
            print(f"Error in sample selection: {str(e)}")
            # Fallback to random selection
            selected_indices = np.random.choice(len(unlabeled_set), num_samples, replace=False)
            selected_samples = [unlabeled_set[i] for i in selected_indices]
            remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
            return selected_samples, remaining_unlabeled