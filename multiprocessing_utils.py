import torch
import torch.multiprocessing as mp
import copy
import numpy as np
import os
import time
from functools import partial

# Set start method for multiprocessing
# 'spawn' is more compatible with CUDA but slower than 'fork'
# 'fork' may cause issues with CUDA in some environments
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

def train_client(client_id, state_dict, unlabeled_set, criterion, 
                 optimizer_config, scheduler_config, dataloaders, 
                 num_epochs, device_id=None):
    """
    Worker function to train a single client in a separate process
    
    Args:
        client_id: ID of the client to train
        state_dict: Initial model state dictionary
        unlabeled_set: List of unlabeled dataset indices
        criterion: Loss function
        optimizer_config: Configuration for optimizer (lr, momentum, etc.)
        scheduler_config: Configuration for scheduler
        dataloaders: Dict of dataloaders
        num_epochs: Number of training epochs
        device_id: GPU device ID to use (if None, use CPU)
    
    Returns:
        dict: Contains trained model state dict, updated optimizer and scheduler states
    """
    # Set device for this process
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    cpu_state_dict = {}
    for key, tensor in state_dict.items():
        cpu_state_dict[key] = tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor


    # Initialize model and load state
    import models.preact_resnet as resnet
    model = resnet.preact_resnet8_cifar(num_classes=10).to(device)
    model.load_state_dict(cpu_state_dict, strict=False)
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optimizer_config['lr'],
        momentum=optimizer_config['momentum'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=scheduler_config['milestones']
    )
    
    # Create a new DataLoader with num_workers=0 to avoid nested multiprocessing
    from torch.utils.data import DataLoader
    
    # Get the dataset and sampler from original dataloader
    original_loader = dataloaders['train-private'][client_id]
    dataset = original_loader.dataset
    sampler = original_loader.sampler
    batch_size = original_loader.batch_size
    
    # Create new dataloader with no worker processes
    local_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # Critical: No multiprocessing in worker process
        pin_memory=True
    )
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for data in local_dataloader:  # Use local dataloader instead
            inputs = data[0].to(device)
            labels = data[1].to(device)
            
            optimizer.zero_grad()
            scores, _ = model(inputs)
            loss = torch.sum(criterion(scores, labels)) / labels.size(0)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    # Return results
    result = {
        'client_id': client_id,
        'state_dict': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in model.state_dict().items()},
        'unlabeled_set': unlabeled_set.copy() if unlabeled_set else None
    }

    model = model.cpu()
    torch.cuda.empty_cache()

    return result

def select_samples_worker(client_id, client_state_dict, server_state_dict, unlabeled_set, 
                          strategy_manager, num_samples, device_id=None):
    """
    Worker function to perform sample selection for a client
    
    Args:
        client_id: ID of the client
        client_state_dict: State dict of the client model
        server_state_dict: State dict of the server model
        unlabeled_set: List of unlabeled dataset indices
        strategy_manager: Strategy manager instance
        num_samples: Number of samples to select
        device_id: GPU device ID to use (if None, use CPU)
    
    Returns:
        dict: Contains selected samples and remaining unlabeled set
    """
    # Set device for this process
    if device_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    
    # Load data
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from data.sampler import SubsetSequentialSampler
    
    # Initialize models
    import models.preact_resnet as resnet
    client_model = resnet.preact_resnet8_cifar(num_classes=10).to(device)
    server_model = resnet.preact_resnet8_cifar(num_classes=10).to(device)
    
    client_model.load_state_dict(client_state_dict, strict=False)
    server_model.load_state_dict(server_state_dict, strict=False)
    
    # Create data loader for this client
    cifar10_select_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    cifar10_select = CIFAR10('data/cifar-10-batches-py', train=True, download=False, 
                             transform=cifar10_select_transform)
    
    unlabeled_loader = DataLoader(
        cifar10_select, 
        batch_size=64,  # Adjust batch size as needed
        sampler=SubsetSequentialSampler(unlabeled_set),
        pin_memory=True,
        num_workers=0
    )
    
    # Select samples using the strategy manager
    selected_samples, remaining_unlabeled = strategy_manager.select_samples(
        client_model,
        server_model,
        unlabeled_loader,
        client_id,
        unlabeled_set,
        num_samples
    )
    
    # Return results
    result = {
        'client_id': client_id,
        'selected_samples': selected_samples,
        'remaining_unlabeled': remaining_unlabeled
    }
    return result

def parallel_train_clients(models, criterion, optimizers, schedulers, dataloaders, 
                           selected_clients_id, num_epochs, num_processes=None):
    """
    Train multiple clients in parallel
    
    Args:
        models: Dict containing client and server models
        criterion: Loss function
        optimizers: Dict containing optimizers for clients
        schedulers: Dict containing schedulers for clients
        dataloaders: Dict containing dataloaders
        selected_clients_id: List of client IDs to train
        num_epochs: Number of training epochs
        num_processes: Number of parallel processes (if None, use CPU count)
    
    Returns:
        dict: Updated models and optimizer states
    """
    if num_processes is None:
        num_processes = min(len(selected_clients_id), mp.cpu_count())
    
    # If only one process, run sequentially to avoid overhead
    if num_processes <= 1 or len(selected_clients_id) == 1:
        results = []
        for c in selected_clients_id:
            result = train_client(
                c, 
                models['clients'][c].state_dict(), 
                None, 
                criterion, 
                {'lr': optimizers['clients'][c].param_groups[0]['lr'], 
                 'momentum': 0.9, 'weight_decay': 5e-4},
                {'milestones': [260]},
                dataloaders,
                num_epochs,
                device_id=0
            )
            results.append(result)
        return results
    
    # Set up multiprocessing pool
    pool = mp.Pool(processes=num_processes)
    
    # Prepare tasks
    tasks = []
    for i, c in enumerate(selected_clients_id):
        # Distribute across available GPUs if multiple are available
        device_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
        
        task = pool.apply_async(
            train_client,
            args=(
                c, 
                models['clients'][c].state_dict(), 
                None, 
                criterion, 
                {'lr': optimizers['clients'][c].param_groups[0]['lr'], 
                 'momentum': 0.9, 'weight_decay': 5e-4},
                {'milestones': [260]},
                dataloaders,
                num_epochs,
                device_id
            )
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = [task.get() for task in tasks]
    
    # Close the pool
    pool.close()
    pool.join()
    
    return results

def parallel_select_samples(models, strategy_manager, unlabeled_set_list, 
                            add, num_processes=None):
    """
    Perform sample selection for multiple clients in parallel
    
    Args:
        models: Dict containing client and server models
        strategy_manager: Strategy manager instance
        unlabeled_set_list: List of unlabeled sets for each client
        add: Number of samples to add for each client
        num_processes: Number of parallel processes (if None, use CPU count)
    
    Returns:
        list: List of results containing selected samples and remaining unlabeled sets
    """
    num_clients = len(models['clients'])
    
    if num_processes is None:
        num_processes = min(num_clients, mp.cpu_count())
    
    # If only one process, run sequentially to avoid overhead
    if num_processes <= 1 or num_clients == 1:
        results = []
        for c in range(num_clients):
            result = select_samples_worker(
                c,
                models['clients'][c].state_dict(),
                models['server'].state_dict(),
                unlabeled_set_list[c],
                strategy_manager,
                add[c],
                device_id=0
            )
            results.append(result)
        return results
    
    # Set up multiprocessing pool
    pool = mp.Pool(processes=num_processes)
    
    # Prepare tasks
    tasks = []
    for i in range(num_clients):
        # Distribute across available GPUs if multiple are available
        device_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
        
        task = pool.apply_async(
            select_samples_worker,
            args=(
                i,
                models['clients'][i].state_dict(),
                models['server'].state_dict(),
                unlabeled_set_list[i],
                strategy_manager,
                add[i],
                device_id
            )
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = [task.get() for task in tasks]
    
    # Close the pool
    pool.close()
    pool.join()
    
    return results