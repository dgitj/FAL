
import random
import numpy as np

class RandomSampler:
    def __init__(self, device):
        self.device = device

    def select_samples(self, model, unlabeled_loader, unlabeled_set, num_samples, seed=None):
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
        # Determine number of samples to select (not more than available)
        num_to_select = min(num_samples, len(unlabeled_set))
        
        if seed is not None:
            # Use numpy's RandomState for more control
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(unlabeled_set), num_to_select, replace=False)
            
            # Select the corresponding samples
            selected_samples = [unlabeled_set[i] for i in indices]
            
        else:    
            # Fallback to standard random.sample for backward compatibility
            selected_samples = random.sample(unlabeled_set, min(num_samples, len(unlabeled_set)))
            # Get remaining samples deterministically
            remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]

        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        return selected_samples, remaining_unlabeled
