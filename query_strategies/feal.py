import torch
import torch.nn.functional as F
import numpy as np
import torch.distributions as dist
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class FEALSampler:
    def __init__(self, device="cuda", n_neighbor=5, cosine=0.85):
        """
        Initializes the Federated Evidential Active Learning (FEAL) sampler.
        
        Args:
            device (str): Device to run the calculations on
            n_neighbor (int): Number of neighbors to consider in diversity check
            cosine (float): Cosine similarity threshold
        """
        self.device = device
        self.n_neighbor = n_neighbor
        self.cosine = cosine


    def compute_discrepancy(self, global_model, local_model, unlabeled_loader):
        """
        Computes uncertainty and feature embeddings from global and local models.

        Args:
            global_model (torch.nn.Module): The global model.
            local_model (torch.nn.Module): The local model.
            unlabeled_loader (DataLoader): CIFAR10-style DataLoader with SubsetSequentialSampler.

        Returns:
            tuple: (global uncertainty, local uncertainty, global uncertainty entropy, 
                   local feature embeddings, original_indices)
        """
        global_model.eval()
        local_model.eval()

        g_u_data_list = torch.tensor([]).to(self.device)
        l_u_data_list = torch.tensor([]).to(self.device)
        g_u_dis_list = torch.tensor([]).to(self.device)
        l_feature_list = torch.tensor([]).to(self.device)
        original_indices = []

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(unlabeled_loader):
                inputs = inputs.to(self.device)

                if hasattr(unlabeled_loader.sampler, 'indices'):
                    start_idx = batch_idx * unlabeled_loader.batch_size
                    end_idx = min((batch_idx + 1) * unlabeled_loader.batch_size, len(unlabeled_loader.sampler.indices))
                    batch_indices = [unlabeled_loader.sampler.indices[i] for i in range(start_idx, end_idx)]
                    original_indices.extend(batch_indices)
                else:
                    batch_indices = list(range(
                        batch_idx * unlabeled_loader.batch_size,
                        min((batch_idx + 1) * unlabeled_loader.batch_size, len(unlabeled_loader.dataset))
                    ))
                    original_indices.extend(batch_indices)

                # Global model uncertainty - Handle tuple return format (scores, features)
                g_output = global_model(inputs)

                if isinstance(g_output, tuple):
                    g_logit, g_features = g_output
                    # Handle if features is a list of block outputs
                    if isinstance(g_features, list):
                        g_feature = g_features[-1]  # Use last block's features
                else:
                    g_logit = g_output
                    g_feature = None

                alpha = F.relu(g_logit) + 1
                total_alpha = torch.sum(alpha, dim=1, keepdim=True)
                dirichlet = dist.Dirichlet(alpha)
                g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
                g_u_dis = dirichlet.entropy()

                # Local model uncertainty - Handle tuple return format
                l_output = local_model(inputs)
                if isinstance(l_output, tuple):
                    l_logit, l_features = l_output
                    # Handle if features is a list of block outputs
                    if isinstance(l_features, list):
                        l_feature = l_features[-1]  # Use last block's features
                else:
                    l_logit = l_output
                    l_feature = None

                alpha = F.relu(l_logit) + 1
                total_alpha = torch.sum(alpha, dim=1, keepdim=True)
                l_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)

                # Collect outputs
                g_u_data_list = torch.cat((g_u_data_list, g_u_data))
                l_u_data_list = torch.cat((l_u_data_list, l_u_data))
                g_u_dis_list = torch.cat((g_u_dis_list, g_u_dis))
                if l_feature is not None:
                    l_feature_list = torch.cat((l_feature_list, l_feature))

        return g_u_data_list, l_u_data_list, g_u_dis_list, l_feature_list, original_indices

    def relaxation(self, u_rank_arg, l_feature_list, neighbor_num, query_num, unlabeled_len, original_indices):
        """
        Ensures diversity in selected samples via neighbor checking.
        
        Args:
            u_rank_arg: Ranked indices based on uncertainty
            l_feature_list: Feature embeddings for each sample
            neighbor_num: Number of neighbors to check
            query_num: Number of samples to select
            unlabeled_len: Length of unlabeled dataset
            original_indices: Original dataset indices of each sample
            
        Returns:
            list: Ranked original indices (not selected yet) from least to most important
        """
        query_flag = torch.zeros(unlabeled_len).to(self.device)
        chosen_idxs = []  # In-batch indices that were chosen
        chosen_orig_indices = []  # Original dataset indices that were chosen

        for i in u_rank_arg:
            if len(chosen_idxs) == query_num:
                break

            cos_sim = pairwise_cosine_similarity(l_feature_list[i:i+1, :], l_feature_list)[0]
            neighbor_arg = torch.argsort(-cos_sim)
            # Use self.cosine instead of undefined cosine variable
            neighbor_arg = neighbor_arg[cos_sim[neighbor_arg] > self.cosine][1:1 + neighbor_num]

            if torch.sum(query_flag[neighbor_arg]) == 0 or len(neighbor_arg) < neighbor_num:
                query_flag[i] = 1
                chosen_idxs.append(i.item())
                chosen_orig_indices.append(original_indices[i.item()])

        # Get remaining indices in correct order
        remaining_idxs = [i for i in range(unlabeled_len) if i not in chosen_idxs]
        remaining_orig_indices = [original_indices[i] for i in remaining_idxs]
        
        ranked_orig_indices = remaining_orig_indices + chosen_orig_indices
        return ranked_orig_indices

    def select_samples(self, global_model, local_model, unlabeled_loader, unlabeled_set, num_samples, seed=None):
        """
        Selects samples with FEAL strategy using proper index mapping.

        Args:
            global_model (torch.nn.Module): The global model.
            local_model (torch.nn.Module): The local model.
            unlabeled_loader (DataLoader): DataLoader with SubsetSequentialSampler.
            unlabeled_set (list): Indices of unlabeled data.
            num_samples (int): Number of samples to select.

        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        num_samples = min(num_samples, len(unlabeled_set))
        unlabeled_len = len(unlabeled_set)

            # Stage 1: Compute uncertainties with explicit index tracking
        g_data, l_data, g_dis, l_features, original_indices = self.compute_discrepancy(
                global_model, local_model, unlabeled_loader
        )
            
        # Verify index tracking was successful
        if len(original_indices) != unlabeled_len or len(g_data) != unlabeled_len:
                 raise ValueError(f"Index tracking mismatch in FEAL. "
                    f"Expected {unlabeled_len}, got {len(original_indices)} indices and {len(g_data)} scores")
                

        # Stage 2: Uncertainty calibration
        if g_dis.max() - g_dis.min() < 1e-10:
            raise ValueError("Global uncertainty range is too small for meaningful calibration")

        u_dis_norm = (g_dis - g_dis.min()) / (g_dis.max() - g_dis.min() + 1e-10) 
        uncertainty = u_dis_norm * (g_data + l_data)

        if uncertainty.numel() == 0:
            raise ValueError("No valid uncertainty scores computed")

        u_rank_arg = torch.argsort(-uncertainty).cpu().numpy()

        # Stage 3: Relaxation for diversity with proper index mapping
        ranked_orig_indices = self.relaxation(
            u_rank_arg=u_rank_arg,
            l_feature_list=l_features,
            neighbor_num=self.n_neighbor,
            query_num=num_samples,
            unlabeled_len=unlabeled_len,
            original_indices=original_indices
        )
            
        # Get actual dataset indices (not just positions in unlabeled_set)
        selected_samples = ranked_orig_indices[-num_samples:]  
        remaining_unlabeled = ranked_orig_indices[:-num_samples]

        if len(set(selected_samples)) != len(selected_samples):
            raise ValueError("Duplicate samples in FEAL selection result")
            
        intersection = set(selected_samples).intersection(set(remaining_unlabeled))
        if intersection:
            raise ValueError(f"{len(intersection)} selected samples still in remaining set - implementation error")
        
        return selected_samples, remaining_unlabeled
            
    