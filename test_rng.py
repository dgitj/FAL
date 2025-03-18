import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from functools import partial

# --- Top-level worker function ---
def seed_worker_fn(base_seed, worker_id):
    """Worker init function that sets the seed for each worker."""
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def get_seed_worker(base_seed):
    """Return a picklable worker initialization function using partial."""
    return partial(seed_worker_fn, base_seed)

# --- Dummy dataset that returns its index ---
class DummyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # Return the data and the index for debugging purposes.
        return self.data[idx], idx

# --- DataLoader test function ---
def test_dataloader_order(labeled_indices, num_workers, base_seed):
    # Use a fixed generator for reproducibility
    g = torch.Generator()
    g.manual_seed(base_seed)
    worker_fn = get_seed_worker(base_seed)
    
    # Create a dummy dataset where data items are simply their indices
    data = list(range(100))
    dataset = DummyDataset(data)
    
    # Create a sampler with the fixed labeled_indices (this controls the order)
    sampler = SubsetRandomSampler(labeled_indices)
    
    # Create the DataLoader with the given num_workers and generator
    loader = DataLoader(dataset, batch_size=10, sampler=sampler,
                        num_workers=num_workers,
                        worker_init_fn=worker_fn,
                        generator=g)
    
    loaded_order = []
    for batch in loader:
        # batch is a tuple: (data_tensor, index_tensor)
        _, indices = batch
        loaded_order.extend(indices.tolist())
    return loaded_order

# --- Main testing block ---
if __name__ == '__main__':
    # Use a fixed list of indices for the sampler (for example, the first 50 indices)
    labeled_indices = list(range(50))
    base_seed = 42

    # Test DataLoader with num_workers=0 (sequential-like)
    print("DataLoader order with num_workers=0 (sequential-like):")
    order_seq = test_dataloader_order(labeled_indices, num_workers=0, base_seed=base_seed)
    print(order_seq)

    # Test DataLoader with num_workers=4 (parallel-like)
    print("\nDataLoader order with num_workers=4 (parallel-like):")
    order_par = test_dataloader_order(labeled_indices, num_workers=4, base_seed=base_seed)
    print(order_par)

    # Compare the two orders
    identical = order_seq == order_par
    print("\nAre the loaded orders identical? ", identical)
