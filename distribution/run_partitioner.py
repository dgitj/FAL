#!/usr/bin/env python3
import argparse
import json
import torch
import numpy as np
from torchvision import datasets
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

# Import the partitioning function
# Assuming the function is saved in a file named dirichlet_partition.py
from data.dirichlet_partitioner import dirichlet_balanced_partition

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Dirichlet partitioning and save results to JSON')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'svhn'],
                        help='Dataset to partition (default: cifar10)')
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients to partition data for (default: 10)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter (default: 0.5)')
    parser.add_argument('--output', type=str, default='partition.json',
                        help='Output JSON file path (default: partition.json)')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to store/load the dataset (default: ./data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load dataset without transformations since we only need labels for partitioning
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'cifar10':
        dataset = CIFAR10(root=args.data_path, train=True, download=True, transform=None)
    elif args.dataset == 'cifar100':
        dataset = CIFAR100(root=args.data_path, train=True, download=True, transform=None)
    elif args.dataset == 'svhn':
        dataset = SVHN(root=args.data_path, split='train', download=True, transform=None)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Run partitioning
    print(f"Partitioning dataset for {args.num_clients} clients with alpha={args.alpha}")
    client_data = dirichlet_balanced_partition(dataset, args.num_clients, args.alpha, args.seed)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = []
    for client_indices in client_data:
        serializable_data.append([int(idx) for idx in client_indices])
    
    # Calculate some statistics for validation
    client_sizes = [len(indices) for indices in serializable_data]
    print(f"Partitioning complete.")
    print(f"Client data sizes: min={min(client_sizes)}, max={max(client_sizes)}, avg={sum(client_sizes)/len(client_sizes):.1f}")
    
    # Save to JSON
    print(f"Saving partitioning to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(serializable_data
        , f, indent=2)
    
    print(f"Done! Partition saved to {args.output}")

if __name__ == "__main__":
    main()