import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
import random
from torchvision import datasets, transforms
from sklearn.metrics import mutual_info_score
import json

# Import the Dirichlet partitioner function
def dirichlet_partition(dataset, num_clients, alpha, balance_size=True, max_variation=0.1, seed=42):
    """
    Partitions dataset into non-IID client datasets using Dirichlet distribution.
    
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
    np.random.seed(seed)
    random.seed(seed)
    
    num_samples = len(dataset)
    num_classes = 10  # Assuming CIFAR-10
    target_size = num_samples // num_clients  # Target size per client
    
    # Calculate acceptable bounds
    lower_bound = int(target_size * (1 - max_variation))
    upper_bound = int(target_size * (1 + max_variation))
    
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
    
    # Balance client sizes if requested
    if balance_size:
        # Total current samples
        current_sizes = [len(client) for client in client_data]
        
        # Redistribute if needed
        while max(current_sizes) - min(current_sizes) > int(target_size * max_variation):
            # Find clients to redistribute from and to
            max_client = np.argmax(current_sizes)
            min_client = np.argmin(current_sizes)
            
            # Move some samples
            move_amount = min((max(current_sizes) - target_size), 
                             (target_size - min(current_sizes)))
            
            # Find indices to move
            indices_to_move = client_data[max_client][:move_amount]
            
            # Update client data
            client_data[max_client] = client_data[max_client][move_amount:]
            client_data[min_client].extend(indices_to_move)
            
            # Update current sizes
            current_sizes = [len(client) for client in client_data]
    
    return client_data

def calculate_non_iid_metrics(partitioned_data, targets, num_clients, num_classes):
    """
    Calculate multiple metrics to quantify non-IID characteristics
    """
    def calculate_entropy(probabilities):
        """Calculate entropy of a probability distribution"""
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    class_entropies = []
    class_concentrations = []
    mutual_infos = []
    kl_divergences = []
    chi_square_results = []
    class_distributions = []
    
    # Global class distribution
    global_dist = np.bincount(targets, minlength=num_classes) / len(targets)
    
    for client_indices in partitioned_data:
        # Client-specific labels
        client_labels = targets[client_indices]
        
        # Compute class distribution
        class_counts = np.bincount(client_labels, minlength=num_classes)
        class_probs = class_counts / len(client_indices)
        
        # Store distributions
        class_distributions.append(class_probs.tolist())
        
        # Entropy of class distribution
        class_entropies.append(calculate_entropy(class_probs))
        class_concentrations.append(np.max(class_probs))
        
        # Mutual Information
        mi = mutual_info_score(client_labels, targets[client_indices])
        mutual_infos.append(mi)
        
        # Chi-square test for uniform distribution
        try:
            chi2, p_value = stats.chisquare(class_counts)
            chi_square_results.append((chi2, p_value))
        except Exception:
            chi_square_results.append((np.nan, np.nan))
        
        # Kullback-Leibler Divergence
        kl_div = np.sum(class_probs * np.log(class_probs / (global_dist + 1e-10) + 1e-10))
        kl_divergences.append(kl_div)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Subplots for different metrics
    metrics_plots = [
        (class_entropies, 'Class Distribution Entropy', 'Entropy'),
        (class_concentrations, 'Class Concentration', 'Max Class Probability'),
        (mutual_infos, 'Mutual Information', 'Mutual Information'),
        (kl_divergences, 'KL Divergence', 'KL Divergence')
    ]
    
    for i, (data, title, xlabel) in enumerate(metrics_plots, 1):
        plt.subplot(2, 2, i)
        plt.hist(data, bins=20)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'non_iid_metrics_alpha{globals().get("alpha", "unknown")}.png')
    plt.close()
    
    # Compile metrics
    metrics = {
        "class_entropy": {
            "mean": float(np.mean(class_entropies)),
            "std": float(np.std(class_entropies)),
            "min": float(np.min(class_entropies)),
            "max": float(np.max(class_entropies))
        },
        "class_concentration": {
            "mean": float(np.mean(class_concentrations)),
            "std": float(np.std(class_concentrations)),
            "min": float(np.min(class_concentrations)),
            "max": float(np.max(class_concentrations))
        },
        "mutual_information": {
            "mean": float(np.mean(mutual_infos)),
            "std": float(np.std(mutual_infos)),
            "min": float(np.min(mutual_infos)),
            "max": float(np.max(mutual_infos))
        },
        "kl_divergence": {
            "mean": float(np.mean(kl_divergences)),
            "std": float(np.std(kl_divergences)),
            "min": float(np.min(kl_divergences)),
            "max": float(np.max(kl_divergences))
        },
        "chi_square": {
            "mean_statistic": float(np.nanmean([r[0] for r in chi_square_results])),
            "mean_p_value": float(np.nanmean([r[1] for r in chi_square_results])),
            "significant_clients": sum(1 for r in chi_square_results if not np.isnan(r[1]) and r[1] < 0.05)
        },
        "class_distributions": class_distributions
    }
    
    return metrics

def run_non_iid_analysis(
    alpha_values=[0.01, 0.1, 1.0, 10.0], 
    num_clients=100,
    num_classes=10,
    seed=42
):
    """
    Run comprehensive non-IID analysis for Dirichlet partitioning
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    targets = np.array(train_dataset.targets)
    
    # Results storage
    results = {}
    
    # Comparative analysis across alpha values
    plt.figure(figsize=(15, 5))
    
    for i, alpha in enumerate(alpha_values, 1):
        # Make alpha global for visualization purposes
        globals()['alpha'] = alpha
        print(f"\nAnalyzing alpha = {alpha}")
        
        # Use the Dirichlet partitioner
        partitioned_data = dirichlet_partition(
            train_dataset, 
            num_clients, 
            alpha, 
            balance_size=True, 
            max_variation=0.1,
            seed=seed
        )
        
        # Compute non-IID metrics
        metrics = calculate_non_iid_metrics(partitioned_data, targets, num_clients, num_classes)
        results[alpha] = metrics
        
        # Print key metrics
        print("Non-IID Metrics Summary:")
        print(f"  Class Concentration: {metrics['class_concentration']['mean']:.4f}")
        print(f"  Mutual Information: {metrics['mutual_information']['mean']:.4f}")
        print(f"  KL Divergence: {metrics['kl_divergence']['mean']:.4f}")
    
    # Comparative visualization
    metrics_to_plot = [
        ('class_concentration', 'mean'), 
        ('mutual_information', 'mean'), 
        ('kl_divergence', 'mean')
    ]
    
    for i, (metric, stat) in enumerate(metrics_to_plot, 1):
        plt.subplot(1, 3, i)
        values = [results[alpha][metric][stat] for alpha in alpha_values]
        plt.plot(alpha_values, values, marker='o')
        plt.xscale('log')
        plt.title(f'{metric.replace("_", " ").title()} ({stat})')
        plt.xlabel('Alpha (log scale)')
        plt.ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('non_iid_alpha_comparison.png')
    plt.close()
    
    # Save detailed results
    with open('non_iid_analysis_results.json', 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    return results

# Main execution
if __name__ == "__main__":
    results = run_non_iid_analysis()
    print("\nAnalysis complete. Results saved to non_iid_analysis_results.json")