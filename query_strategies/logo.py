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
        Extracts feature embeddings from the local model.

        Args:
            model (torch.nn.Module): Local model.
            unlabeled_loader (DataLoader): DataLoader for unlabeled data.

        Returns:
            torch.Tensor: All embeddings concatenated.
        """
        model.eval()
        embeddings = []

        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.to(self.device)
                _, features = model(inputs)
                embeddings.append(features.cpu())

        return torch.cat(embeddings, dim=0)

    def macro_micro_clustering(self, model_server, unlabeled_loader, unlabeled_set, embeddings, num_samples):
        """
        Performs LoGo's macro (clustering) and micro (uncertainty-based selection) steps.

        Args:
            model_server (torch.nn.Module): Global model.
            unlabeled_loader (DataLoader): Full unlabeled DataLoader.
            unlabeled_set (list): Indices of the unlabeled data.
            embeddings (torch.Tensor): Feature embeddings from the local model.
            num_samples (int): Number of samples to select.

        Returns:
            list: Selected sample indices.
        """
        # Macro step: K-Means clustering
        kmeans = KMeans(n_clusters=num_samples, random_state=0)
        cluster_labels = kmeans.fit_predict(embeddings.numpy())

        # Group indices by cluster
        cluster_dict = {i: [] for i in range(num_samples)}
        for idx, cluster_id in zip(unlabeled_set, cluster_labels):
            cluster_dict[cluster_id].append(idx)

        # Micro step: Uncertainty-based selection within each cluster
        selected_samples = []
        for cluster_id, cluster_idxs in cluster_dict.items():
            if not cluster_idxs:
                continue

            cluster_subset = torch.utils.data.Subset(unlabeled_loader.dataset, cluster_idxs)
            cluster_loader = torch.utils.data.DataLoader(
                cluster_subset, batch_size=64, shuffle=False
            )

            uncertainties = []
            with torch.no_grad():
                for inputs, _ in cluster_loader:
                    inputs = inputs.to(self.device)
                    outputs, _ = model_server(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    log_probs = torch.log(probs + 1e-12)
                    entropy = -torch.sum(probs * log_probs, dim=1)
                    uncertainties.extend(entropy.cpu().numpy())

            most_uncertain_idx = cluster_idxs[np.argmax(uncertainties)]
            selected_samples.append(most_uncertain_idx)

        return selected_samples

    def select_samples(self, model, model_server, unlabeled_loader, c, unlabeled_set, num_samples):
        """
        Selects samples using the LoGo strategy.

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
        embeddings = self.extract_embeddings(model, unlabeled_loader)
        selected_samples = self.macro_micro_clustering(
            model_server, unlabeled_loader, unlabeled_set, embeddings, num_samples
        )
        remaining_unlabeled = list(set(unlabeled_set) - set(selected_samples))

        return selected_samples, remaining_unlabeled
