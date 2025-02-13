import json
import numpy as np
from torchvision import datasets, transforms

def dirichlet_partition_cifar10(dataset, num_clients, alpha):
    """
    Partitions CIFAR-10 dataset into non-IID client datasets using Dirichlet distribution.
    Returns a nested list where each sublist contains client-specific sample indices.
    """

    num_samples = len(dataset)  # 50,000 training samples
    num_classes = 10  # CIFAR-10 has 10 classes
    labels = np.array(dataset.targets)  # CIFAR-10 labels

    # Create a dictionary to store class-wise indices
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    # Initialize an empty list for each client
    client_data = [[] for _ in range(num_clients)]

    # Dirichlet distribution to determine data split
    class_shares = np.random.dirichlet([alpha] * num_clients, num_classes)

    total_assigned = 0

    # Assign data samples to clients based on Dirichlet allocation
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)  # Shuffle class-specific indices
        proportions = (class_shares[c] * len(indices)).astype(int)

        start = 0
        for i in range(num_clients):
            client_data[i].extend(indices[start: start + proportions[i]])
            start += proportions[i]
            total_assigned += proportions[i]

    # Handle missing samples
    missing_samples = num_samples - total_assigned  # Samples not assigned due to rounding
    if missing_samples > 0:
        print(f"Fixing {missing_samples} missing samples...")
        all_remaining_indices = np.concatenate(list(class_indices.values()))
        np.random.shuffle(all_remaining_indices)

        for i in range(missing_samples):
            client_data[i % num_clients].append(int(all_remaining_indices[i]))



    # Convert NumPy int64 to Python int for JSON serialization
    formatted_data = [[int(idx) for idx in client_samples] for client_samples in client_data]

    return formatted_data

# Example usage
if __name__ == "__main__":
    # Load CIFAR-10 dataset
    data_dir = "../data/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)

    num_clients = 80  # Number of partitions
    alpha = 0.1  # Dirichlet concentration parameter

    # Generate partitions
    partitioned_data = dirichlet_partition_cifar10(train_dataset, num_clients, alpha)

    # Verify the total sample count
    total_samples = sum(len(client) for client in partitioned_data)
    print(f"Total assigned samples: {total_samples}")  # Should print 50000

    # Save to JSON file
    filename = "alpha0-1_cifar10_80clients.json"
    with open(filename, "w") as json_file:
        json.dump(partitioned_data, json_file, indent=4)

    print(f"Partitioned data saved to {filename}.")

