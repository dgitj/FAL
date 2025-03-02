import torch
import torch.nn.functional as F
import numpy as np
import torch.distributions as dist
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class FEALSampler:
    def __init__(self, device="cuda", n_neighbor=5, cosine=0.85):
        """
        Initializes the Federated Evidential Active Learning (FEAL) sampler.
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
            unlabeled_loader (DataLoader): CIFAR10-style DataLoader (inputs, labels).

        Returns:
            tuple: (global uncertainty, local uncertainty, global uncertainty entropy, local feature embeddings)
        """
        global_model.eval()
        local_model.eval()

        g_u_data_list = torch.tensor([]).to(self.device)
        l_u_data_list = torch.tensor([]).to(self.device)
        g_u_dis_list = torch.tensor([]).to(self.device)
        l_feature_list = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.to(self.device)

                # Global model uncertainty
                g_logit, _, _, _ = global_model(inputs)
                alpha = F.relu(g_logit) + 1
                total_alpha = torch.sum(alpha, dim=1, keepdim=True)
                dirichlet = dist.Dirichlet(alpha)
                g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
                g_u_dis = dirichlet.entropy()

                # Local model uncertainty
                l_logit, _, _, block_features = local_model(inputs)
                l_feature = block_features[-1]  # ✅ FIXED: Use features directly
                alpha = F.relu(l_logit) + 1
                total_alpha = torch.sum(alpha, dim=1, keepdim=True)
                l_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)

                # Collect outputs
                g_u_data_list = torch.cat((g_u_data_list, g_u_data))
                l_u_data_list = torch.cat((l_u_data_list, l_u_data))
                g_u_dis_list = torch.cat((g_u_dis_list, g_u_dis))
                l_feature_list = torch.cat((l_feature_list, l_feature))

        return g_u_data_list, l_u_data_list, g_u_dis_list, l_feature_list

    def relaxation(self, u_rank_arg, l_feature_list, neighbor_num, query_num, unlabeled_len):
        """
        Ensures diversity in selected samples via neighbor checking.
        """
        query_flag = torch.zeros(unlabeled_len).to(self.device)
        chosen_idx = []

        for i in u_rank_arg:
            if len(chosen_idx) == query_num:
                break

            cos_sim = pairwise_cosine_similarity(l_feature_list[i:i+1, :], l_feature_list)[0]
            neighbor_arg = torch.argsort(-cos_sim)
            neighbor_arg = neighbor_arg[cos_sim[neighbor_arg] > cosine][1:1 + neighbor_num]

            if torch.sum(query_flag[neighbor_arg]) == 0 or len(neighbor_arg) < neighbor_num:
                query_flag[i] = 1
                chosen_idx.append(i.item())

        remain_idx = list(set(range(unlabeled_len)) - set(chosen_idx))
        rank_arg = remain_idx + chosen_idx
        return rank_arg

    def select_samples(self, global_model, local_model, unlabeled_loader, unlabeled_set, num_samples):
        """
        Selects samples with FEAL strategy.

        Args:
            global_model (torch.nn.Module): The global model.
            local_model (torch.nn.Module): The local model.
            unlabeled_loader (DataLoader): CIFAR10-style DataLoader (inputs, labels).
            unlabeled_set (list): Indices of unlabeled data.
            num_samples (int): Number of samples to select.
            args: Args object containing `n_neighbor` and `cosine`.

        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        unlabeled_len = len(unlabeled_set)

        # Stage 1: Compute uncertainties
        g_data, l_data, g_dis, l_features = self.compute_discrepancy(global_model, local_model, unlabeled_loader)

        # Stage 2: Uncertainty calibration
        u_dis_norm = (g_dis - g_dis.min()) / (g_dis.max() - g_dis.min())
        uncertainty = u_dis_norm * (g_data + l_data)
        u_rank_arg = torch.argsort(-uncertainty).cpu().numpy()

        # Stage 3: Relaxation for diversity
        rank_arg = self.relaxation(
            u_rank_arg=u_rank_arg,
            l_feature_list=l_features,
            neighbor_num=args.n_neighbor,
            query_num=num_samples,
            unlabeled_len=unlabeled_len,
            cosine=args.cosine
        )

        selected_samples = [unlabeled_set[i] for i in rank_arg[:num_samples]]
        remaining_unlabeled = [unlabeled_set[i] for i in rank_arg[num_samples:]]

        return selected_samples, remaining_unlabeled
