import random

class RandomSampler:
    def __init__(self, device):
        self.device = device

    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples):
        """
        Randomly selects a subset of samples from the unlabeled pool.
        
        Args:
            model (torch.nn.Module): Unused but kept for consistency with other strategies.
            unlabeled_loader (DataLoader): Unused since we're selecting randomly.
            unlabeled_set (list): List of indices of the unlabeled samples.
            num_samples (int): Number of samples to select.

        Returns:
            selected_samples (list): Indices of randomly selected samples.
            remaining_unlabeled (list): Indices of the remaining unlabeled samples.
        """
        selected_samples = random.sample(unlabeled_set, min(num_samples, len(unlabeled_set)))
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]

        return selected_samples, remaining_unlabeled
