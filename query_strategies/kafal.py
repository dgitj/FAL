import numpy as np
import torch
import torch.nn.functional as F

class KAFALSampler:
    def __init__(self, loss_weight_list, device="cuda"):
        """
        Initializes the KAFL discrepancy-based sampler.

        Args:
            loss_weight_list (list of torch.Tensor): List of class-specific weights for each client.
            device (str): Device to run calculations on (e.g., 'cuda' or 'cpu').
        """
        self.loss_weight_list = loss_weight_list
        self.device = device

    def compute_discrepancy(self, model, model_server, unlabeled_loader, c):
        """
        Computes the discrepancy between the client's model and the server's model.

        Args:
            model (torch.nn.Module): The client model.
            model_server (torch.nn.Module): The global server model.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            c (int): ID of the client.

        Returns:
            torch.Tensor: A tensor of discrepancy scores for each sample.
        """
        model.eval()
        model_server.eval()
        discrepancy = torch.tensor([]).to(self.device)

        with torch.no_grad():
            for (inputs, labels) in unlabeled_loader:
                inputs = inputs.to(self.device)
                
                scores, features = model(inputs)
                scores_t, _ = model_server(inputs)

                # Intensify specialized knowledge
                spc = self.loss_weight_list[c]
                spc = spc.unsqueeze(0).expand(inputs.size(0), -1)
                
                const = 1
                scores += const * spc.log()
                scores_t += const * spc.log()

                # Compute symmetric KL divergence
                kl_client_server = F.kl_div(
                    F.log_softmax(scores, -1),
                    F.softmax(scores_t, -1),
                    reduction='none'
                ).sum(1)

                kl_server_client = F.kl_div(
                    F.log_softmax(scores_t, -1),
                    F.softmax(scores, -1),
                    reduction='none'
                ).sum(1)
                
                discrepancy_batch = 0.5 * (kl_client_server + kl_server_client)
                discrepancy = torch.cat((discrepancy, discrepancy_batch), 0)

        return discrepancy.cpu()

    def select_samples(self, model, model_server, unlabeled_loader, c, unlabeled_set, num_samples):
        """
        Selects samples with the highest discrepancy scores.

        Args:
            model (torch.nn.Module): The client model.
            model_server (torch.nn.Module): The global server model.
            unlabeled_loader (DataLoader): DataLoader for the unlabeled data.
            c (int): ID of the client.
            unlabeled_set (list): List of indices corresponding to the unlabeled data.
            num_samples (int): Number of samples to select.

        Returns:
            tuple: Selected samples and remaining unlabeled samples.
        """
        discrepancy = self.compute_discrepancy(model, model_server, unlabeled_loader, c)
        arg = np.argsort(discrepancy)

        # Select the top `num_samples` samples with the highest discrepancy
        selected_samples = list(torch.tensor(unlabeled_set)[arg][-num_samples:].numpy())
        remaining_unlabeled = list(torch.tensor(unlabeled_set)[arg][:-num_samples].numpy())

        return selected_samples, remaining_unlabeled
