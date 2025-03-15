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

    def add_noise_to_weights(self, model, seed=None):
        """
        Adds Gaussian noise to model weights.
        """
        with torch.no_grad():
            # Create a deterministic generator if seed is provided
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)


            for param in model.parameters():
                if param.requires_grad:
                    # Calculate normalization factor to keep relative noise scale consistent
                    param_norm = param.norm()
                    if param_norm > 0:  # Avoid division by zero
                        # Generate noise with the same generator (or without for backward compatibility)
                        if generator is not None:
                            noise1 = torch.randn_like(param, generator=generator)
                            # Need a different seed for the second randn to avoid division by zero
                            gen2 = torch.Generator(device=self.device)
                            gen2.manual_seed(seed + param_idx + 10000)
                            noise2 = torch.randn_like(param, generator=gen2)
                            noise_norm = torch.norm(noise2)
                        else:
                            noise1 = torch.randn_like(param)
                            noise_norm = torch.norm(torch.randn_like(param))
                            
                        if noise_norm > 0:  # Prevent division by zero
                            noise = noise1 * self.noise_scale * param_norm / noise_norm
                            param.add_(noise)

    def compute_uncertainty(self, model, unlabeled_loader, seed=None):
        """
        Computes feature deviations before and after adding noise.

        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.

        Returns:
            torch.Tensor: Uncertainty scores for the samples.
        """
        model.eval()
        
        try:
            # Get original outputs and features
            outputs, features = self.get_all_outputs(model, unlabeled_loader)
            if features is None:
                error_msg = "Feature extraction failed - model may not return feature embeddings"
                print(error_msg)
                raise ValueError(error_msg)
                
            
            # Initialize difference tensor
            diffs = torch.zeros_like(features).to(self.device)

            # Apply noise multiple times and measure deviation
            for i in range(self.num_sampling):
                # Create a deep copy of the model to avoid modifying the original
                noisy_model = copy.deepcopy(model).to(self.device)
                # Use a deterministic seed derived from the base seed
                noise_seed = None if seed is None else seed + i
                self.add_noise_to_weights(noisy_model)
                noisy_model.eval()
                
                # Get outputs from noisy model
                _, noisy_features = self.get_all_outputs(noisy_model, unlabeled_loader)
                if noisy_features is None:
                    continue
                    
                # Calculate absolute difference
                diff_k = noisy_features - features
                diffs += diff_k.abs()

            # Normalize by number of successful noise iterations
            if self.num_sampling > 0:
                diffs = diffs / self.num_sampling
                
            # Return mean difference across feature dimensions
            return diffs.mean(dim=1)
        
        except Exception as e:
            error_msg = f"Error in compute_uncertainty: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg) from e

    def get_all_outputs(self, model, dataloader):
        """
        Runs the model on all samples and returns outputs and feature embeddings.

        Args:
            model (torch.nn.Module): The model used for predictions.
            dataloader (DataLoader): The dataset loader.

        Returns:
            tuple: (outputs, features) - Probability distributions and feature embeddings
        """
        model.eval()
        outputs_list = []
        features_list = []

        with torch.no_grad():
            for batch in dataloader:
                # Handle different DataLoader formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)

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
                        return torch.cat(outputs_list, dim=0), None
                        
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue

        if not outputs_list or not features_list:
            error_msg = "No valid outputs or features collected"
            print(error_msg)
            raise ValueError(error_msg)
            
        try:
            return torch.cat(outputs_list, dim=0), torch.cat(features_list, dim=0)
        except Exception as e:
            print(f"Error concatenating results: {str(e)}")
            return None, None

    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples, seed=None):
        """
        Selects the most uncertain samples based on feature deviation.

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
        
        try:
            # Compute uncertainty scores
            uncertainty = self.compute_uncertainty(model, unlabeled_loader)
            
            if uncertainty is None or len(uncertainty) != len(unlabeled_set):
                error_msg = f"Uncertainty computation failed or returned incorrect size. Expected {len(unlabeled_set)}, got {len(uncertainty) if uncertainty is not None else 'None'}"
                print(error_msg)
                raise ValueError(error_msg)
                
            # Select indices with highest uncertainty
            sorted_indices = torch.argsort(uncertainty, descending=True).cpu().numpy()
            selected_indices = sorted_indices[:num_samples]
            
            # Map to actual sample indices
            selected_samples = [unlabeled_set[i] for i in selected_indices]
            remaining_unlabeled = [idx for i, idx in enumerate(unlabeled_set) if i not in selected_indices]
            
            return selected_samples, remaining_unlabeled
            
        except Exception as e:
            error_msg = f"Error in sample selection: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg) from e