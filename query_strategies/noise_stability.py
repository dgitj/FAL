import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances

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
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.noise_scale * param.norm() / torch.norm(torch.randn_like(param))
                    param.add_(noise)

    def compute_uncertainty(self, model, unlabeled_loader):
        """
        Computes feature deviations before and after adding noise.

        Args:
            model (torch.nn.Module): The model used for predictions.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.

        Returns:
            torch.Tensor: Uncertainty scores for the samples.
        """
        model.eval()
        use_feature = True  # Use feature representations instead of softmax scores

        # Get original outputs
        outputs = self.get_all_outputs(model, unlabeled_loader, use_feature)

        diffs = torch.zeros_like(outputs).to(self.device)

        # Apply noise multiple times and measure deviation
        for _ in range(self.num_sampling):
            noisy_model = copy.deepcopy(model).to(self.device)
            self.add_noise_to_weights(noisy_model)
            noisy_model.eval()
            outputs_noisy = self.get_all_outputs(noisy_model, unlabeled_loader, use_feature)

            diff_k = outputs_noisy - outputs
            diffs += diff_k.abs()

        return diffs.mean(dim=1)  # Aggregate uncertainty scores

    def get_all_outputs(self, model, dataloader, use_feature=False):
        """
        Runs the model on all samples and returns feature embeddings or softmax outputs.

        Args:
            model (torch.nn.Module): The model used for predictions.
            dataloader (DataLoader): The dataset loader.
            use_feature (bool): Whether to return feature embeddings instead of probabilities.

        Returns:
            torch.Tensor: Model outputs.
        """
        model.eval()
        outputs = []

        with torch.no_grad():
            print("🔹 DEBUG: Starting to process batches...")
            
            for batch_idx, batch in enumerate(dataloader):
                print(f"🔹 Processing batch {batch_idx + 1}...")

                # Ensure the batch has the right number of elements
                if len(batch) == 3:
                    inputs, _, _ = batch  # (inputs, labels, indices)
                elif len(batch) == 2:
                    inputs, _ = batch  # (inputs, labels)
                else:
                    print(f"❌ Unexpected batch format: {batch}")
                    raise ValueError(f"Unexpected number of elements in batch: {len(batch)}")

                inputs = inputs.to(self.device)

                # Run model forward pass
                out, fea = model(inputs)

                # Debugging outputs
                if fea is None:
                    print(f"❌ ERROR: Model returned None for features in batch {batch_idx + 1}")
                    continue
                if out is None:
                    print(f"❌ ERROR: Model returned None for output in batch {batch_idx + 1}")
                    continue

                print(f"✅ Batch {batch_idx + 1} processed, feature shape: {fea.shape}")

                outputs.append(fea if use_feature else F.softmax(out, dim=1))

        if not outputs:
            print("❌ ERROR: No outputs collected. Dataloader might be empty or the model is failing.")
            raise RuntimeError("get_all_outputs() did not process any data. Check the dataset and model.")

        return torch.cat(outputs, dim=0)


    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples):
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
        uncertainty = self.compute_uncertainty(model, unlabeled_loader)
        sorted_indices = torch.argsort(uncertainty, descending=True)

        selected_samples = [unlabeled_set[idx] for idx in sorted_indices[:num_samples]]
        remaining_unlabeled = [unlabeled_set[idx] for idx in sorted_indices[num_samples:]]

        return selected_samples, remaining_unlabeled
