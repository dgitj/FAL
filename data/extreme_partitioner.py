import numpy as np
import json
import torchvision
import os

def create_extreme_class_disjoint_partition(dataset, num_clients, num_classes, samples_per_client=None):
    """
    Create a partition where each client primarily has data from 1-2 classes,
    but all clients have the same number of samples.
    
    Args:
        dataset: The dataset to partition
        num_clients: Number of clients
        num_classes: Number of classes in the dataset
        samples_per_client: Number of samples each client should have (if None, will be calculated)
    """
    # Total number of samples
    total_samples = len(dataset)
    
    # Determine number of samples per client if not specified
    if samples_per_client is None:
        samples_per_client = total_samples // num_clients
    
    print(f"Target samples per client: {samples_per_client}")
    
    # Sort data by class
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Assign primary class to each client
    client_data_indices = [[] for _ in range(num_clients)]
    primary_classes = np.random.permutation(num_classes)
    
    # First phase: Assign primary classes (approximately 80-90% of client's data)
    for i in range(num_clients):
        primary_class = primary_classes[i % num_classes]
        primary_samples_target = int(0.85 * samples_per_client)  # Target 85% from primary class
        
        # Check if enough samples are available
        class_data = class_indices[primary_class]
        available_samples = len(class_data)
        
        # Take what we can, up to our target
        take_samples = min(primary_samples_target, available_samples)
        
        # Select samples without replacement
        if take_samples > 0:
            selected_indices = np.random.choice(class_data, take_samples, replace=False)
            client_data_indices[i].extend(selected_indices.tolist())
            
            # Remove selected indices from available pool
            class_indices[primary_class] = [idx for idx in class_data if idx not in selected_indices]
    
    # Second phase: Fill remaining slots with secondary classes
    for i in range(num_clients):
        remaining_slots = samples_per_client - len(client_data_indices[i])
        
        if remaining_slots <= 0:
            continue
            
        # Get primary class for this client
        primary_class = primary_classes[i % num_classes]
        
        # Choose 1-3 secondary classes
        available_classes = [c for c in range(num_classes) if c != primary_class]
        num_secondary = min(3, len(available_classes))
        secondary_classes = np.random.choice(available_classes, num_secondary, replace=False)
        
        # Calculate samples per secondary class
        samples_per_secondary = remaining_slots // num_secondary
        
        for j, secondary_class in enumerate(secondary_classes):
            # For the last class, take all remaining samples
            if j == num_secondary - 1:
                samples_to_take = remaining_slots
            else:
                samples_to_take = samples_per_secondary
                
            # Check available samples for this class
            class_data = class_indices[secondary_class]
            available_samples = len(class_data)
            
            # Take what we can
            take_samples = min(samples_to_take, available_samples)
            
            if take_samples > 0:
                selected_indices = np.random.choice(class_data, take_samples, replace=False)
                client_data_indices[i].extend(selected_indices.tolist())
                remaining_slots -= take_samples
                
                # Remove selected indices from available pool
                class_indices[secondary_class] = [idx for idx in class_data if idx not in selected_indices]
    
    # Final phase: If any client still doesn't have enough samples, take from what's left
    for i in range(num_clients):
        remaining_slots = samples_per_client - len(client_data_indices[i])
        
        if remaining_slots <= 0:
            continue
            
        # Flatten all remaining class indices
        remaining_indices = []
        for class_data in class_indices:
            remaining_indices.extend(class_data)
        
        if len(remaining_indices) > 0:
            # Take what we can
            take_samples = min(remaining_slots, len(remaining_indices))
            
            if take_samples > 0:
                selected_indices = np.random.choice(remaining_indices, take_samples, replace=False)
                client_data_indices[i].extend(selected_indices.tolist())
                
                # Remove selected indices from the pool
                for idx in selected_indices:
                    for c in range(num_classes):
                        if idx in class_indices[c]:
                            class_indices[c].remove(idx)
    
    return client_data_indices

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    num_clients = 10
    num_classes = 10  # For CIFAR-10
    save_path = "extreme_partition.json"
    
    # Load CIFAR-10 dataset (without transform, just for accessing labels)
    print("Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    
    # Create extreme class disjoint partition with balanced client sizes
    print(f"Creating extreme class disjoint partition for {num_clients} clients...")
    client_data_indices = create_extreme_class_disjoint_partition(
        train_dataset, num_clients, num_classes
    )
    
    # Print statistics about the partition
    print("\nPartition statistics:")
    for i, indices in enumerate(client_data_indices):
        # Get class distribution for this client
        client_labels = [train_dataset[idx][1] for idx in indices]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        
        print(f"Client {i}: {len(indices)} samples")
        for label, count in zip(unique_labels, counts):
            print(f"  - Class {label}: {count} samples ({count/len(indices)*100:.1f}%)")
    
    # Save indices to JSON file
    print(f"\nSaving partition to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump(client_data_indices, f, cls=NumpyEncoder)
    
    print("Done!")

if __name__ == "__main__":
    main()