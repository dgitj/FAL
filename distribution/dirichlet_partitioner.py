import torch
import random
import numpy as np
import math
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

def dirichlet_balanced_partition(dataset, num_clients, alpha, seed=42):
    """
    Balanced Dirichlet partitioner that creates equal-sized client datasets
    while preserving non-IID characteristics.
    
    Args:
        dataset: PyTorch dataset (CIFAR10, CIFAR100, or SVHN)
        num_clients: Number of clients to partition for
        alpha: Concentration parameter for Dirichlet distribution
        seed: Random seed for reproducibility
    
    Returns:
        List of lists containing sample indices for each client
    """

      # Debugging: Print dataset type
    print(f"Dataset type: {type(dataset)}")
    print(f"CIFAR10 match: {isinstance(dataset, CIFAR10)}")
    print(f"CIFAR100 match: {isinstance(dataset, CIFAR100)}")
    
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Get number of classes from dataset
    if isinstance(dataset, CIFAR10):
        num_classes = 10
    elif isinstance(dataset, CIFAR100):
        num_classes = 100
    elif isinstance(dataset, SVHN):
        num_classes = 10
    else:
        # Try to infer number of classes
        if hasattr(dataset, 'classes'):
            num_classes = len(dataset.classes)
        else:
            print("Warning: Could not determine number of classes. Assuming 10.")
            num_classes = 10
    
    # Initialize client data storage
    clients_data = {}
    for i in range(num_clients):
        clients_data[i] = []

    # Get dataset labels
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, "train_labels"):
        labels = np.array(dataset.train_labels)
    elif hasattr(dataset, "labels"):  # For SVHN
        labels = np.array(dataset.labels)
    else:
        raise ValueError("Dataset has no recognizable label attribute")
    
    # Find unique labels and create a mapping from label to index
    unique_labels = np.unique(labels)
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    actual_num_classes = len(unique_labels)
    
    if actual_num_classes != num_classes:
        print(f"Warning: Expected {num_classes} classes but found {actual_num_classes} unique labels.")
        num_classes = actual_num_classes
    
    # Organize dataset by class with consistent string conversion
    total_data = {}
    data_num = np.zeros(num_classes)
    
    # First create empty lists for each unique label
    for label in unique_labels:
        total_data[str(int(label))] = []  # Consistent string conversion
    
    # Count samples per class and store indices
    for idx in range(len(dataset)):
        label = labels[idx]
        label_str = str(int(label))  # Consistent string conversion
        total_data[label_str].append(idx)
        data_num[label_to_index[label]] += 1

    # Track per-client class counts
    clients_data_num = {}
    for client in range(num_clients):
        clients_data_num[client] = [0] * num_classes
    
    # Generate Dirichlet distribution
    diri_dis = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(num_classes))
    sample = torch.cat([diri_dis.sample().unsqueeze(0) for _ in range(num_clients)], 0)

    # Balance the matrix to ensure proper row and column sums
    rsum = sample.sum(1)
    csum = sample.sum(0)
    epsilon = min(1, num_clients / num_classes, num_classes / num_clients) / 1000

    if alpha < 10:
        # For highly non-IID scenarios (small alpha)
        r, c = 1, num_clients / num_classes
        iteration = 0
        while (torch.any(rsum <= r - epsilon) or torch.any(csum <= c - epsilon)) and iteration < 1000:
            sample /= sample.sum(0)
            sample /= sample.sum(1).unsqueeze(1)
            rsum = sample.sum(1)
            csum = sample.sum(0)
            iteration += 1
    else:
        # For more IID scenarios (large alpha)
        r, c = num_classes / num_clients, 1
        iteration = 0
        while (torch.any(abs(rsum - r) >= epsilon) or torch.any(abs(csum - c) >= epsilon)) and iteration < 1000:
            sample = sample / sample.sum(1).unsqueeze(1)
            sample /= sample.sum(0)
            rsum = sample.sum(1)
            csum = sample.sum(0)
            iteration += 1
    
    # Calculate sample counts based on Dirichlet distribution
    x = sample * torch.tensor(data_num)
    x = torch.ceil(x).long()
    x = torch.where(x <= 1, 0, x+1) if alpha < 10 else torch.where(x <= 1, 0, x)
    
    print('Dataset total num', len(dataset))
    print('Total dataset class num', data_num)

    # Distribute samples to clients
    if alpha < 10:
        # Handle highly non-IID case
        remain = np.inf
        nums = math.ceil(len(dataset) / num_clients)  # Target size per client
        i = 0
        while remain != 0 and i < 100:
            i += 1
            for client_idx in clients_data.keys():
                for label_str in total_data.keys():
                    # Get the original label and its index in our mapping
                    label = int(label_str)
                    label_idx = label_to_index.get(label, 0)  # Default to 0 if not found
                    
                    # Ensure label_idx is in range for x tensor
                    if label_idx < num_classes:
                        sample_count = x[client_idx, label_idx].item()
                        if sample_count > 0 and len(total_data[label_str]) > 0:
                            # Take the minimum of what's requested and what's available
                            tmp_set = random.sample(total_data[label_str], 
                                          min(len(total_data[label_str]), sample_count))
                            
                            # Ensure we don't exceed target size
                            if len(clients_data[client_idx]) + len(tmp_set) > nums:
                                tmp_set = tmp_set[:nums-len(clients_data[client_idx])]

                            clients_data[client_idx] += tmp_set
                            clients_data_num[client_idx][label_idx] += len(tmp_set)

                            # Remove selected samples from available pool
                            total_data[label_str] = list(set(total_data[label_str])-set(tmp_set))   

            remain = sum([len(d) for _, d in total_data.items()])
                
        # Ensure equal size for all clients
        client_sizes = np.array([sum(clients_data_num[k]) for k in clients_data_num.keys()])
        index = np.where(client_sizes < nums)[0]
        
        for client_idx in index:
            n = nums - len(clients_data[client_idx])
            add = 0
            for label_str in total_data.keys():
                if n <= add or len(total_data[label_str]) == 0:
                    continue
                tmp_set = total_data[label_str][:min(n-add, len(total_data[label_str]))]
                
                clients_data[client_idx] += tmp_set
                label = int(label_str)
                label_idx = label_to_index.get(label, 0)
                clients_data_num[client_idx][label_idx] += len(tmp_set)
                total_data[label_str] = list(set(total_data[label_str])-set(tmp_set))  
                
                add += len(tmp_set)
    else:
        # Handle more IID case
        # Create label index mapping for the cumsum tensor
        cumsum = x.T.cumsum(dim=1)
        
        for label_str, data in total_data.items():
            label = int(label_str)
            label_idx = label_to_index.get(label, 0)
            
            if len(data) == 0:
                continue
                
            if label_idx < len(cumsum):
                cum = list(cumsum[label_idx].numpy())
                # Ensure cum values are valid split points
                cum = [min(v, len(data)) for v in cum]
                cum = sorted(list(set(cum)))
                if len(cum) == 0 or cum[-1] != len(data):
                    cum.append(len(data))
                
                # Handle edge case where cum might be problematic
                try:
                    tmp = np.split(np.array(data), cum)[:-1]  # Skip last empty chunk
                except ValueError:
                    # Fall back to even split
                    chunk_size = len(data) // num_clients
                    tmp = [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_clients)]
                
                for client_idx, client_chunk in enumerate(tmp):
                    if client_idx >= num_clients:
                        break
                    clients_data[client_idx] += list(client_chunk)
                    clients_data_num[client_idx][label_idx] += len(client_chunk)

    # Print statistics
    client_sizes = [sum(clients_data_num[k]) for k in clients_data_num.keys()]
    print('Client data sizes:', client_sizes)
    print(f'Min client size: {min(client_sizes)}, Max client size: {max(client_sizes)}')
    
    # Calculate class distribution statistics
    class_concentrations = []
    for client_id in range(num_clients):
        class_counts = clients_data_num[client_id]
        total = sum(class_counts)
        if total > 0:  # Avoid division by zero
            class_percentages = [count / total for count in class_counts]
            max_class_percentage = max(class_percentages)
            class_concentrations.append(max_class_percentage)
    
    avg_concentration = sum(class_concentrations) / len(class_concentrations)
    print(f'Average class concentration: {avg_concentration:.4f}')
    print(f'Max class concentration: {max(class_concentrations):.4f}')
    
    # Convert to the format expected by original code (list of lists with int indices)
    formatted_data = []
    for i in range(num_clients):
        formatted_data.append([int(idx) for idx in clients_data[i]])
    
    return formatted_data