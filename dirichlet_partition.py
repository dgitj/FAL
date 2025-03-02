import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
import os
import random
from torchvision import datasets, transforms

def set_seed(seed):
    """Set seed for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def dirichlet_partition(dataset, num_clients, alpha, balance_size=True, max_variation=0.1, seed=42):
    """
    Partitions dataset into non-IID client datasets using Dirichlet distribution.
    With bounded variation balancing to preserve non-IID characteristics.
    
    Args:
        dataset: Dataset to partition
        num_clients: Number of clients to partition data for
        alpha: Concentration parameter for Dirichlet distribution
        balance_size: Whether to ensure clients have bounded dataset sizes
        max_variation: Maximum allowed variation from target size (as fraction)
        seed: Random seed for reproducibility
    
    Returns:
        A list where each element is a list of indices for a client
    """
    # Set random seed
    set_seed(seed)
    
    num_samples = len(dataset)
    num_classes = 10  # Assuming CIFAR-10
    target_size = num_samples // num_clients  # Target size per client
    
    # Calculate acceptable bounds
    lower_bound = int(target_size * (1 - max_variation))
    upper_bound = int(target_size * (1 + max_variation))
    
    print(f"Target size: {target_size} | Acceptable range: {lower_bound} to {upper_bound}")
    
    # Get dataset labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "train_labels"):
        labels = np.array(dataset.train_labels)
    else:
        labels = np.array([target for _, target in dataset])
    
    # Create a dictionary to store class-wise indices
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    
    # Initial Dirichlet allocation without balancing
    client_data = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        indices = class_indices[c].copy()
        np.random.shuffle(indices)
        
        # Get Dirichlet distribution for this class
        class_shares = np.random.dirichlet([alpha] * num_clients)
        
        # Distribute samples according to Dirichlet shares
        proportions = np.floor(class_shares * len(indices)).astype(int)
        remainder = len(indices) - np.sum(proportions)
        
        if remainder > 0:
            additional = np.random.choice(num_clients, remainder, replace=False)
            for idx in additional:
                proportions[idx] += 1
        
        # Assign samples to clients
        start = 0
        for i in range(num_clients):
            end = start + proportions[i]
            client_data[i].extend(indices[start:end].tolist())
            start = end
    
    # Calculate initial sizes
    initial_sizes = [len(data) for data in client_data]
    print(f"Initial sizes: min={min(initial_sizes)}, max={max(initial_sizes)}, target={target_size}")
    print(f"Initial coefficient of variation: {np.std(initial_sizes)/np.mean(initial_sizes):.4f}")
    
    if balance_size:
        # Create a transfer pool
        transfer_pool = []
        
        # Find clients that exceed upper bound and remove excess
        for i in range(num_clients):
            if len(client_data[i]) > upper_bound:
                excess = len(client_data[i]) - upper_bound
                print(f"Client {i} has {len(client_data[i])} samples, removing {excess}")
                
                # Get class distribution
                client_labels = labels[client_data[i]]
                class_counts = {c: np.sum(client_labels == c) for c in range(num_classes)}
                class_percent = {c: count/len(client_data[i]) for c, count in class_counts.items()}
                
                # Sort classes by representation (most represented first)
                sorted_classes = sorted(range(num_classes), key=lambda c: class_percent.get(c, 0), reverse=True)
                
                # Select samples to remove, starting with most represented classes
                # This helps preserve the non-IID characteristics
                indices_to_remove = []
                for c in sorted_classes:
                    # Find indices of this class
                    class_indices = [idx for idx in client_data[i] if labels[idx] == c]
                    # Determine how many to remove (proportional to representation)
                    to_remove = min(int(excess * class_percent.get(c, 0) * 1.5), len(class_indices))
                    
                    if to_remove > 0:
                        selected = random.sample(class_indices, to_remove)
                        indices_to_remove.extend(selected)
                        
                    if len(indices_to_remove) >= excess:
                        indices_to_remove = indices_to_remove[:excess]
                        break
                
                # If we still need more, select randomly from remaining
                if len(indices_to_remove) < excess:
                    remaining = [idx for idx in client_data[i] if idx not in indices_to_remove]
                    additional = random.sample(remaining, excess - len(indices_to_remove))
                    indices_to_remove.extend(additional)
                
                # Move to transfer pool
                transfer_pool.extend(indices_to_remove)
                
                # Remove from client
                client_data[i] = [idx for idx in client_data[i] if idx not in set(indices_to_remove)]
        
        # Shuffle transfer pool
        random.shuffle(transfer_pool)
        print(f"Transfer pool size: {len(transfer_pool)}")
        
        # Find clients below lower bound and add samples
        for i in range(num_clients):
            if len(client_data[i]) < lower_bound and transfer_pool:
                deficit = lower_bound - len(client_data[i])
                print(f"Client {i} has {len(client_data[i])} samples, adding {min(deficit, len(transfer_pool))}")
                
                # Get current class distribution
                if client_data[i]:  # If client has samples
                    client_labels = labels[client_data[i]]
                    class_counts = {c: np.sum(client_labels == c) for c in range(num_classes)}
                    total = len(client_data[i])
                    class_percent = {c: class_counts.get(c, 0)/total for c in range(num_classes)}
                else:  # Empty client (unlikely but possible)
                    class_percent = {c: 0 for c in range(num_classes)}
                
                # Sort classes by representation (least represented first)
                sorted_classes = sorted(range(num_classes), key=lambda c: class_percent.get(c, 0))
                
                # Organize transfer pool by class
                transfer_by_class = {c: [] for c in range(num_classes)}
                for idx in transfer_pool:
                    transfer_by_class[labels[idx]].append(idx)
                
                # Select samples to add, prioritizing under-represented classes
                samples_to_add = []
                remaining_deficit = deficit
                
                for c in sorted_classes:
                    # Try to get samples from this class
                    available = transfer_by_class[c]
                    # Calculate how many to add from this class
                    desired = min(remaining_deficit, len(available))
                    
                    if desired > 0:
                        selected = available[:desired]
                        samples_to_add.extend(selected)
                        # Remove selected samples from pool
                        for idx in selected:
                            transfer_by_class[c].remove(idx)
                        
                        remaining_deficit -= desired
                    
                    if remaining_deficit == 0:
                        break
                
                # If we still need more, take any available samples
                if remaining_deficit > 0:
                    remaining_pool = [idx for sublist in transfer_by_class.values() for idx in sublist]
                    additional = remaining_pool[:remaining_deficit]
                    samples_to_add.extend(additional)
                
                # Add to client
                client_data[i].extend(samples_to_add)
                
                # Remove from transfer pool
                transfer_pool = [idx for idx in transfer_pool if idx not in set(samples_to_add)]
        
        # Handle any remaining samples in transfer pool
        if transfer_pool:
            print(f"Distributing {len(transfer_pool)} remaining samples...")
            # Find clients that can still accept samples (below upper bound)
            eligible_clients = [i for i in range(num_clients) if len(client_data[i]) < upper_bound]
            
            if eligible_clients:
                # Sort by current size (ascending)
                eligible_clients.sort(key=lambda i: len(client_data[i]))
                
                for idx in transfer_pool:
                    # Find smallest eligible client that won't exceed upper bound
                    for i in eligible_clients:
                        if len(client_data[i]) < upper_bound:
                            client_data[i].append(idx)
                            # Reorder if this client is no longer the smallest
                            eligible_clients.sort(key=lambda i: len(client_data[i]))
                            break
    
    # Calculate final sizes
    final_sizes = [len(data) for data in client_data]
    print(f"\nFinal sizes: min={min(final_sizes)}, max={max(final_sizes)}, target={target_size}")
    print(f"Final coefficient of variation: {np.std(final_sizes)/np.mean(final_sizes):.4f}")
    
    # Convert to list of integers
    formatted_data = [[int(idx) for idx in client] for client in client_data]
    
    return formatted_data

def analyze_partitions(partitioned_data, targets, num_clients, num_classes, alpha):
    """Perform statistical analysis on the partitioned data."""
    # 1. Size distribution analysis
    client_sizes = [len(client_indices) for client_indices in partitioned_data]
    target_size = sum(client_sizes) // num_clients
    
    print("\n=== Size Distribution Analysis ===")
    print(f"Client data sizes: {client_sizes}")
    print(f"Min size: {min(client_sizes)}, Max size: {max(client_sizes)}")
    print(f"Mean size: {np.mean(client_sizes):.2f}, Median size: {np.median(client_sizes):.2f}")
    print(f"Std dev: {np.std(client_sizes):.2f}, CV: {np.std(client_sizes)/np.mean(client_sizes):.4f}")
    print(f"Max deviation from target: {max(abs(np.array(client_sizes) - target_size)):.2f} samples")
    print(f"Max percentage deviation: {100 * max(abs(np.array(client_sizes) - target_size))/target_size:.2f}%")
    
    # 2. Class distribution analysis
    client_class_counts = []
    client_class_percentages = []
    
    print("\n=== Class Distribution Analysis ===")
    for client_id, client_indices in enumerate(partitioned_data):
        client_labels = targets[client_indices]
        class_counts = np.zeros(num_classes)
        
        for c in range(num_classes):
            class_counts[c] = np.sum(client_labels == c)
            
        client_class_counts.append(class_counts)
        client_class_percentages.append(class_counts / len(client_indices))
        
        # Print summary for each client
        print(f"Client {client_id}: Size={len(client_indices)}, Most common class: {np.argmax(class_counts)} ({100*np.max(class_counts)/len(client_indices):.1f}%)")
    
    # Convert to numpy arrays for easier analysis
    client_class_counts = np.array(client_class_counts)
    client_class_percentages = np.array(client_class_percentages)
    
    # Calculate class imbalance metrics
    class_concentration = np.max(client_class_percentages, axis=1)
    print(f"\nClass concentration (% of most common class): {100*class_concentration.mean():.2f}% avg, {100*class_concentration.min():.2f}% min, {100*class_concentration.max():.2f}% max")
    
    # 3. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 3.1 Client size distribution
    axes[0, 0].bar(range(num_clients), client_sizes)
    axes[0, 0].axhline(y=np.mean(client_sizes), color='r', linestyle='--', label=f'Mean: {np.mean(client_sizes):.1f}')
    axes[0, 0].axhline(y=target_size * 1.1, color='g', linestyle='--', label='110% of target')
    axes[0, 0].axhline(y=target_size * 0.9, color='g', linestyle='--', label='90% of target')
    axes[0, 0].set_title('Client Dataset Sizes')
    axes[0, 0].set_xlabel('Client ID')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].legend()
    
    # 3.2 Class distribution heatmap
    sns.heatmap(client_class_percentages, annot=True, fmt='.2f', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title(f'Class Distribution (α={alpha})')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Client ID')
    
    # 3.3 Distribution of class proportions for each class
    axes[1, 0].boxplot([client_class_percentages[:, c] for c in range(num_classes)])
    axes[1, 0].set_title('Distribution of Class Proportions')
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('Proportion')
    
    # 3.4 Size variation from target
    deviations = np.array(client_sizes) - target_size
    axes[1, 1].bar(range(num_clients), deviations)
    axes[1, 1].axhline(y=0, color='k', linestyle='-')
    axes[1, 1].axhline(y=target_size * 0.1, color='g', linestyle='--', label='+10% Target')
    axes[1, 1].axhline(y=-target_size * 0.1, color='g', linestyle='--', label='-10% Target')
    axes[1, 1].set_title('Client Size Deviation from Target')
    axes[1, 1].set_xlabel('Client ID')
    axes[1, 1].set_ylabel('Deviation (samples)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'partition_analysis_alpha{alpha}_clients{num_clients}.png')
    plt.close()
    
    # Return all statistics for metadata
    stats_data = {
        "client_sizes": client_sizes,
        "size_mean": float(np.mean(client_sizes)),
        "size_median": float(np.median(client_sizes)),
        "size_std": float(np.std(client_sizes)),
        "size_cv": float(np.std(client_sizes)/np.mean(client_sizes)),
        "size_max_deviation": float(max(abs(deviations))),
        "size_max_deviation_percent": float(100 * max(abs(deviations))/target_size),
        "class_concentration_mean": float(class_concentration.mean()),
        "class_concentration_min": float(class_concentration.min()),
        "class_concentration_max": float(class_concentration.max()),
        "class_distribution": client_class_percentages.tolist()
    }
    
    return stats_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate and analyze Dirichlet partitions')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha parameter for Dirichlet distribution')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--balance', action='store_true', default=True, help='Balance client dataset sizes')
    parser.add_argument('--no-balance', action='store_false', dest='balance', help='Do not balance client dataset sizes')
    parser.add_argument('--max_variation', type=float, default=0.1, help='Maximum allowed variation from target size')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing dataset')
    parser.add_argument('--compare_alphas', action='store_true', help='Compare different alpha values')
    args = parser.parse_args()
    
    # Set master seed
    set_seed(args.seed)
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    print(f"Loading CIFAR-10 dataset from {args.data_dir}")
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    
    # Get targets
    targets = np.array(train_dataset.targets)
    num_classes = 10
    
    # Generate partitions
    print(f"Generating partitions with parameters: num_clients={args.num_clients}, alpha={args.alpha}, seed={args.seed}, balance={args.balance}, max_variation={args.max_variation}")
    partitioned_data = dirichlet_partition(
        train_dataset, 
        args.num_clients, 
        args.alpha, 
        args.balance, 
        args.max_variation,
        args.seed
    )
    
    # Analyze the partitions
    stats_data = analyze_partitions(partitioned_data, targets, args.num_clients, num_classes, args.alpha)
    
    # Create metadata
    metadata = {
        "algorithm": "dirichlet",
        "alpha": args.alpha,
        "num_clients": args.num_clients,
        "balanced": args.balance,
        "max_variation": args.max_variation,
        "seed": args.seed,
        "dataset": "cifar10",
        "total_samples": len(train_dataset),
        "statistics": stats_data
    }
    
    # Save to JSON file

    filename = f"alpha{args.alpha}_cifar10_{args.num_clients}clients_var{args.max_variation}_seed{args.seed}.json"
    with open(filename, "w") as json_file:
        json.dump(partitioned_data, json_file)
    
    print(f"\nPartitioned data saved to {filename}")
    print(f"Visualization saved to partition_analysis_alpha{args.alpha}_clients{args.num_clients}.png")
    
    # Compare different alpha values if requested
    if args.compare_alphas:
        print("\n=== Comparison with different alpha values ===")
        alpha_values = [0.01, 0.1, 1.0, 10.0]
        class_concentrations = []
        size_cvs = []
        
        for a in alpha_values:
            if a == args.alpha:
                # Skip computation for already analyzed alpha
                class_concentrations.append(stats_data["class_concentration_mean"])
                size_cvs.append(stats_data["size_cv"])
                continue
                
            print(f"\nAnalyzing alpha={a}")
            part_data = dirichlet_partition(
                train_dataset, 
                args.num_clients, 
                a, 
                args.balance, 
                args.max_variation,
                args.seed
            )
            
            stats = analyze_partitions(part_data, targets, args.num_clients, num_classes, a)
            class_concentrations.append(stats["class_concentration_mean"])
            size_cvs.append(stats["size_cv"])
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot alpha vs class concentration
        ax1.plot(alpha_values, [100*conc for conc in class_concentrations], 'o-')
        ax1.set_xscale('log')
        ax1.set_xlabel('Alpha (log scale)')
        ax1.set_ylabel('Average Class Concentration (%)')
        ax1.set_title('Effect of Alpha on Class Concentration')
        ax1.grid(True)
        
        # Plot alpha vs size variation
        ax2.plot(alpha_values, size_cvs, 'o-')
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha (log scale)')
        ax2.set_ylabel('Size Coefficient of Variation')
        ax2.set_title('Effect of Alpha on Size Variation')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('alpha_comparison.png')
        
        print("\nClass concentration and size variation by alpha:")
        for a, conc, cv in zip(alpha_values, class_concentrations, size_cvs):
            print(f"Alpha={a}: {100*conc:.2f}% concentration, {cv:.4f} size CV")

if __name__ == "__main__":
    main()