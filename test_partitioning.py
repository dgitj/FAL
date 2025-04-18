"""
Test script to verify that the Dirichlet partitioning is reproducible
and consistent between SSL-FL and FAL implementations.
"""

import torch
import numpy as np
import random
import json
import os
from torchvision import datasets

# Import the FAL partitioner
from data.dirichlet_partitioner import dirichlet_balanced_partition

def test_partitioning_reproducibility():
    """Test if partitioning is reproducible with the same seed."""
    # Configuration
    num_clients = 5
    alpha = 0.5  # Non-IID degree
    seed = 42    # Fixed seed for reproducibility
    
    # Load CIFAR10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=None  # No transform needed for this test
    )
    
    print(f"Testing partitioning reproducibility with seed={seed}, alpha={alpha}")
    
    # First partitioning
    print("\nRunning first partitioning...")
    client_indices1 = dirichlet_balanced_partition(
        train_dataset, num_clients, alpha, seed
    )
    
    # Second partitioning with same parameters
    print("\nRunning second partitioning...")
    client_indices2 = dirichlet_balanced_partition(
        train_dataset, num_clients, alpha, seed
    )
    
    # Check if client indices are identical
    indices_match = all(
        np.array_equal(sorted(client_indices1[i]), sorted(client_indices2[i]))
        for i in range(num_clients)
    )
    
    print(f"\nIndices match between runs: {indices_match}")
    
    # Calculate class distributions for comparison
    labels = np.array(train_dataset.targets)
    true_distributions = []
    
    for client_id in range(num_clients):
        client_labels = labels[client_indices1[client_id]]
        class_counts = np.bincount(client_labels, minlength=10)
        true_distributions.append(class_counts / class_counts.sum())
    
    # Save metadata for comparison with SSL-FL
    test_output = {
        "alpha": alpha,
        "seed": seed,
        "num_clients": num_clients,
        "client_sizes": [len(indices) for indices in client_indices1],
        "class_distributions": [dist.tolist() for dist in true_distributions],
        # Save a subset of indices for verification (first 5 indices for each client)
        "sample_indices": [client_indices1[i][:5] for i in range(num_clients)]
    }
    
    # Save to JSON file
    output_path = "fal_partition_test.json"
    with open(output_path, "w") as f:
        json.dump(test_output, f, indent=2)
    
    print(f"\nTest metadata saved to {output_path}")
    print(f"Compare this with the SSL-FL test output")

if __name__ == "__main__":
    test_partitioning_reproducibility()
