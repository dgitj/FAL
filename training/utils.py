"""
Utility functions for federated active learning.
Provides helper functions for reproducibility, data handling, and model operations.
"""

import os
import random
import functools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.sampler import SubsetSequentialSampler
import time
from datetime import datetime


def set_all_seeds(seed):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed (int): Seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker_fn(base_seed, worker_id):
    """
    Sets unique seed for each dataloader worker.
    
    Args:
        base_seed (int): Base seed value to derive worker seed from
        worker_id (int): Worker ID
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


def get_seed_worker(base_seed):
    """
    Creates a worker initialization function with the given base seed.
    
    Args:
        base_seed (int): Base seed for worker initialization
        
    Returns:
        function: Worker initialization function for DataLoader
    """
    return functools.partial(seed_worker_fn, base_seed)


def read_data(dataloader):
    """
    Creates an infinite iterator over a dataloader.
    Useful for continuously feeding data in training loops.
    
    Args:
        dataloader (DataLoader): Source dataloader
        
    Returns:
        generator: Infinite iterator over dataloader contents
    """
    while True:
        for data in dataloader:
            yield data


def create_data_loaders(dataset, labeled_indices, unlabeled_indices, batch_size, seed):
    """
    Create dataloaders for labeled and unlabeled data with reproducible behavior.
    
    Args:
        dataset (torch.utils.data.Dataset): Source dataset
        labeled_indices (list): Indices of labeled samples
        unlabeled_indices (list): Indices of unlabeled samples
        batch_size (int): Batch size for dataloaders
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (labeled_loader, unlabeled_loader)
    """
    # Create seed-dependent worker and generator
    worker_init_fn = get_seed_worker(seed)
    
    g_labeled = torch.Generator()
    g_labeled.manual_seed(seed + 10000)
    
    g_unlabeled = torch.Generator()
    g_unlabeled.manual_seed(seed + 20000)
    
    # Create dataloaders
    labeled_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=SubsetRandomSampler(labeled_indices),
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=g_labeled,
        pin_memory=True
    )
    
    unlabeled_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=SubsetSequentialSampler(unlabeled_indices),
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=g_unlabeled,
        pin_memory=True
    )
    
    return labeled_loader, unlabeled_loader


def get_device():
    """
    Determine the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():   
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def calculate_model_distance(local_model, global_model):
    """
    Calculate Euclidean distance between model parameters.
    
    Args:
        local_model (nn.Module): Local model
        global_model (nn.Module): Global model
        
    Returns:
        float: Euclidean distance between model parameters
    """
    distance = 0.0
    local_params = dict(local_model.named_parameters())
    global_params = dict(global_model.named_parameters())
    
    for name, param in global_params.items():
        if name in local_params:
            distance += torch.norm(param - local_params[name]).item() ** 2
            
    return np.sqrt(distance)


def create_experiment_name(config):
    """
    Generate a descriptive name for an experiment based on config.
    
    Args:
        config: Configuration module with parameters
        
    Returns:
        str: Experiment name
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.ACTIVE_LEARNING_STRATEGY}_c{config.CLIENTS}_a{config.ALPHA}_{timestamp}"


def log_config(config):
    """
    Print configuration parameters in a readable format.
    
    Args:
        config: Configuration module
    """
    print("\n===== Experiment Configuration =====")
    for key, value in vars(config).items():
        if not key.startswith('__') and key.isupper():
            print(f"{key}: {value}")
    print("===================================\n")


def create_results_dir(dir_name="results"):
    """
    Create directory for storing experiment results.
    
    Args:
        dir_name (str): Name of directory to create
        
    Returns:
        str: Path to created directory
    """
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def calculate_class_distribution(indices, labels, num_classes=10):
    """
    Calculate class distribution for a subset of data.
    
    Args:
        indices (list): Indices of samples to analyze
        labels (list or numpy.ndarray): Labels for all samples
        num_classes (int): Number of classes
        
    Returns:
        numpy.ndarray: Array with count of samples per class
    """
    distribution = np.zeros(num_classes)
    for idx in indices:
        label = labels[idx]
        distribution[label] += 1
    return distribution


class Timer:
    """Simple timer for measuring execution time of code blocks."""
    
    def __init__(self, name="Timer"):
        """Initialize timer with optional name."""
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        """Start timing when entering context."""
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        """Print elapsed time when exiting context."""
        elapsed = time.time() - self.start_time
        print(f"{self.name}: {elapsed:.4f} seconds")


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    
    Args:
        model (nn.Module): Model to modify
        flag (bool): Value for requires_grad
    """
    for param in model.parameters():
        param.requires_grad = flag


def count_parameters(model):
    """
    Count number of trainable parameters in a model.
    
    Args:
        model (nn.Module): Model to analyze
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_size(model):
    """
    Calculate the size of a model in terms of parameters and bytes.
    
    Args:
        model (nn.Module): PyTorch model to analyze
        
    Returns:
        tuple: (num_parameters, num_bytes)
            - num_parameters (int): Total number of parameters
            - num_bytes (int): Total size in bytes
    """
    total_params = 0
    total_bytes = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        bytes_per_param = param.element_size()
        total_bytes += num_params * bytes_per_param
    
    return total_params, total_bytes