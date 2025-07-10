"""
Main entry point for federated active learning experiments.
Implements federated learning with various active learning sampling strategies.
"""

import os
import random
import argparse
import json
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler


# Import models
import models.preact_resnet as resnet
import models.preact_resnet_mnist as resnet_mnist
from training.federated_ssl_trainer import perform_federated_ssl_pretraining
from models.ssl_models import create_model_with_pretrained_encoder_cifar, create_model_with_pretrained_encoder_mnist


# Import data utilities
from data.dirichlet_partitioner import dirichlet_balanced_partition
from data.sampler import SubsetSequentialSampler

# Import training module components
from training.trainer import FederatedTrainer
from training.evaluation import evaluate_model, evaluate_per_class_accuracy
from training.utils import (
    set_all_seeds, get_seed_worker, log_config, get_device, create_results_dir
)

# Import active learning strategies
from query_strategies.strategy_manager import StrategyManager

# Import logger
from analysis.logger_strategy import FederatedALLogger

# Import configuration
import config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Active Learning')
    parser.add_argument('--strategy', type=str, help='Active learning strategy to use')
    parser.add_argument('--cycles', type=int, help='Number of active learning cycles')
    parser.add_argument('--clients', type=int, help='Number of federated clients')
    parser.add_argument('--alpha', type=float, help='Dirichlet partition non-IID level')
    parser.add_argument('--budget', type=int, help='Active learning budget per cycle')
    parser.add_argument('--base', type=int, help='Initial labeled set size')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'SVHN', 'CIFAR100', 'MNIST'], help='Dataset to use')
    parser.add_argument('--save-checkpoints', action='store_true', help='Enable automatic checkpoint saving')
    parser.add_argument('--checkpoint-frequency', type=int, default=1, help='Save checkpoints every N cycles')
    
    return parser.parse_args()


def load_datasets():
    """Load and prepare datasets (CIFAR10, SVHN, CIFAR100, or MNIST)."""
    from torchvision.datasets import CIFAR10, SVHN, CIFAR100, MNIST
    import torchvision.transforms as T

    dataset_name = config.DATASET

    if dataset_name == "CIFAR10":
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        dataset_dir = config.DATA_ROOT
        
        train_dataset = CIFAR10(dataset_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(dataset_dir, train=False, download=True, transform=test_transform)
        select_dataset = CIFAR10(dataset_dir, train=True, download=True, transform=test_transform)
        
    elif dataset_name == "SVHN":
        train_transform = T.Compose([
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
        ])

        dataset_dir = config.DATA_ROOT
        
        train_dataset = SVHN(dataset_dir, split='train', download=True, transform=train_transform)
        test_dataset = SVHN(dataset_dir, split='test', download=True, transform=test_transform)
        select_dataset = SVHN(dataset_dir, split='train', download=True, transform=test_transform)
        
    elif dataset_name == "CIFAR100":
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

        dataset_dir = config.DATA_ROOT
        
        train_dataset = CIFAR100(dataset_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(dataset_dir, train=False, download=True, transform=test_transform)
        select_dataset = CIFAR100(dataset_dir, train=True, download=True, transform=test_transform)
        
    elif dataset_name == "MNIST":
        train_transform = T.Compose([
            T.RandomCrop(size=28, padding=2),  
            T.ToTensor(),
            T.Normalize([0.1307], [0.3081])  
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.1307], [0.3081])
        ])

        dataset_dir = config.DATA_ROOT
        
        train_dataset = MNIST(dataset_dir, train=True, download=True, transform=train_transform)
        test_dataset = MNIST(dataset_dir, train=False, download=True, transform=test_transform)
        select_dataset = MNIST(dataset_dir, train=True, download=True, transform=test_transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    return train_dataset, test_dataset, select_dataset


def create_test_loader(dataset, trial_seed, batch_size=config.BATCH):
    """Create DataLoader for test data."""
    test_generator = torch.Generator()
    test_generator.manual_seed(trial_seed)
    test_worker_init_fn = get_seed_worker(trial_seed + 50000)
    
    test_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        worker_init_fn=test_worker_init_fn, 
        generator=test_generator,
        pin_memory=True
    )
    
    return test_loader

def create_val_loader(dataset, trial_seed, indices=None, batch_size=config.BATCH):
    """Create DataLoader for validation data."""
    val_generator = torch.Generator()
    val_generator.manual_seed(trial_seed + 10000)
    val_worker_init_fn = get_seed_worker(trial_seed + 60000)
    
    if indices is not None:
        sampler = SubsetRandomSampler(indices)
        val_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=val_worker_init_fn, 
            generator=val_generator,
            pin_memory=True
        )
    else:
        val_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            worker_init_fn=val_worker_init_fn, 
            generator=val_generator,
            pin_memory=True
        )
    
    return val_loader


def main():
    """Main function to run federated active learning experiments."""
    # Parse arguments
    args = parse_arguments()
    
    # Create checkpoints directory if checkpoints are enabled
    if args.save_checkpoints:
        os.makedirs('checkpoints', exist_ok=True)
    
    if args.strategy:
        config.ACTIVE_LEARNING_STRATEGY = args.strategy
        print(f"Using strategy: {config.ACTIVE_LEARNING_STRATEGY}")
    
    # Add dataset selection
    if args.dataset:
        config.DATASET = args.dataset
        if config.DATASET == "CIFAR10":
            config.DATA_ROOT = 'data/cifar-10-batches-py'
            config.NUM_CLASSES = 10
            config.NUM_TRAIN = 50000
        elif config.DATASET == "SVHN":
            config.DATA_ROOT = 'data/svhn'
            config.NUM_CLASSES = 10
            config.NUM_TRAIN = 73257  
        elif config.DATASET == "CIFAR100":
            config.DATA_ROOT = 'data/cifar-100-python'
            config.NUM_CLASSES = 100
            config.NUM_TRAIN = 50000
        elif config.DATASET == "MNIST":
            config.DATA_ROOT = 'data/mnist'
            config.NUM_CLASSES = 10
            config.NUM_TRAIN = 60000  
        print(f"Using dataset: {config.DATASET} with {config.NUM_CLASSES} classes and {config.NUM_TRAIN} training samples")
    
    if args.cycles:
        config.CYCLES = args.cycles
        print(f"Setting cycles to: {config.CYCLES}")
    
    if args.clients:
        config.CLIENTS = args.clients
        print(f"Setting number of clients to: {config.CLIENTS}")
    
    if args.alpha is not None:
        config.ALPHA = args.alpha
        print(f"Setting Dirichlet alpha to: {config.ALPHA}")
    
    if args.budget:
        config.BUDGET = args.budget
        print(f"Setting budget to: {config.BUDGET}")
    
    if args.base:
        config.BASE = args.base
        print(f"Setting base size to: {config.BASE}")
    
    if args.seed:
        config.SEED = args.seed
        print(f"Setting random seed to: {config.SEED}")
    
    # Log configuration
    log_config(config)
    
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    
    # Load datasets
    cifar10_train, cifar10_test, cifar10_select = load_datasets()
    
    # Create test transform for SSL feature testing
    import torchvision.transforms as T
    if config.DATASET == "CIFAR10":
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
    elif config.DATASET == "SVHN":
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
        ])
    elif config.DATASET == "CIFAR100":
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])
    elif config.DATASET == "MNIST":
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.1307], [0.3081])
        ])
    else:
        test_transform = T.Compose([T.ToTensor()])
    
    # Prepare for trials
    accuracies = [[] for _ in range(config.TRIALS)]
    
    # Extract all labels
    indices = list(range(len(cifar10_train)))
    if config.DATASET == "CIFAR10":
        id2lab = [cifar10_train[id][1] for id in indices]
    elif config.DATASET == "SVHN":
        id2lab = [cifar10_train.labels[id] for id in indices]
    elif config.DATASET == "MNIST":
        id2lab = [cifar10_train[id][1] for id in indices]
    else:
        id2lab = [cifar10_train[id][1] for id in indices]  
    id2lab = np.array(id2lab)
    
    # Run trials
    for trial in range(config.TRIALS):
        trial_seed = config.SEED + config.TRIAL_SEED_OFFSET * (trial + 1)
        set_all_seeds(trial_seed)
        
        print(f"\n=== Trial {trial+1}/{config.TRIALS} (Seed: {trial_seed}) ===\n")
        print(f"Generating Dirichlet partition with alpha {config.ALPHA}, seed {trial_seed} for {config.CLIENTS} clients...")
        
        # Create non-IID data partitioning
        data_splits = dirichlet_balanced_partition(cifar10_train, config.CLIENTS, alpha=config.ALPHA, seed=trial_seed)
        
        # SSL Pre-training if enabled
        if config.USE_SSL_PRETRAIN:
            print("\n" + "="*70)
            print("=== Starting Federated SSL Pre-training ===")
            print("="*70)
            
            # Get the base dataset without augmentations for SSL
            # (SSL will apply its own augmentations)
            if config.DATASET == "CIFAR10":
                from torchvision.datasets import CIFAR10
                ssl_dataset = CIFAR10(config.DATA_ROOT, train=True, download=False, transform=None)
            elif config.DATASET == "MNIST":
                from torchvision.datasets import MNIST
                ssl_dataset = MNIST(config.DATA_ROOT, train=True, download=False, transform=None)
            # Add other datasets as needed
            
            # Perform federated SSL pre-training
            ssl_pretrained_encoder = perform_federated_ssl_pretraining(
                data_splits=data_splits,
                config=config,
                device=device,
                trial_seed=trial_seed,
                base_dataset=ssl_dataset
            )

             # Test SSL features quality
            print("\n=== Testing SSL Feature Quality ===")
            ssl_pretrained_encoder.eval()
            with torch.no_grad():
                # Get a few random images from the dataset
                test_indices = [0, 100, 200, 300, 400]
                
                features = []
                for idx in test_indices:
                    # Get image from test dataset
                    img, _ = cifar10_test[idx]
                    
                    # Handle different image formats
                    if isinstance(img, torch.Tensor):
                        # Image is already a tensor (likely already transformed)
                        if img.dim() == 3:  # CHW format
                            img_tensor = img
                        else:
                            raise ValueError(f"Unexpected tensor shape: {img.shape}")
                    else:
                        # Image is PIL or numpy, apply transform
                        img_tensor = test_transform(img)
                    
                    # Ensure tensor is on correct device and has batch dimension
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    feat = ssl_pretrained_encoder(img_tensor)
                    features.append(feat)
                    print(f"Image {idx} - Feature shape: {feat.shape}, norm: {feat.norm().item():.4f}")
                
                # Check feature diversity
                similarities = []
                for i in range(len(features)):
                    for j in range(i+1, len(features)):
                        sim = F.cosine_similarity(features[i], features[j])
                        similarities.append(sim.item())
                
                print(f"Average feature similarity: {np.mean(similarities):.4f}")
                print(f"Feature norm mean: {np.mean([f.norm().item() for f in features]):.4f}")
                print(f"Feature norm std: {np.std([f.norm().item() for f in features]):.4f}")
                
                if np.mean(similarities) > 0.95:
                    print("WARNING: Features are too similar - possible feature collapse!")
                elif np.mean(similarities) < 0.3:
                    print("SUCCESS: Features are diverse - SSL is working!")
                else:
                    print("OK: Features show moderate diversity")

            
            # Create model with pre-trained encoder
            if config.DATASET == "MNIST":
                base_model = create_model_with_pretrained_encoder_mnist(
                    ssl_pretrained_encoder,
                    num_classes=config.NUM_CLASSES
                )
            else:
                base_model = create_model_with_pretrained_encoder_cifar(
                    ssl_pretrained_encoder,
                    num_classes=config.NUM_CLASSES
                )
            
            print("SSL Pre-training completed! Using pre-trained features.\n")
        else:
            # Original random initialization
            if config.DATASET == "MNIST":
                base_model = resnet_mnist.preact_resnet8_mnist(num_classes=config.NUM_CLASSES)
            else:
                base_model = resnet.preact_resnet8_cifar(num_classes=config.NUM_CLASSES)



        # Initialize logger
        logger = FederatedALLogger(
            strategy_name=config.ACTIVE_LEARNING_STRATEGY,
            num_clients=config.CLIENTS,
            num_classes=config.NUM_CLASSES,
            trial_id=trial+1 
        )
        
        # Setup for clients
        labeled_set_list = []
        unlabeled_set_list = []
        private_train_loaders = []
        private_unlab_loaders = []
        num_classes = config.NUM_CLASSES
        
        print('Query Strategy:', config.ACTIVE_LEARNING_STRATEGY)
        
        strategy_params = {
            'strategy_name': config.ACTIVE_LEARNING_STRATEGY,
            'loss_weight_list': [],  
            'device': device
        }
        
        # Prepare client data
        if config.DATASET == "MNIST":
            resnet8 = resnet_mnist.preact_resnet8_mnist(num_classes=num_classes)
        else:
            resnet8 = resnet.preact_resnet8_cifar(num_classes=num_classes)
        client_models = []
        data_list = []
        total_data_num = [len(data_splits[c]) for c in range(config.CLIENTS)]
        total_data_num = np.array(total_data_num)
        
        # Calculate base and budget sizes
        base = np.ceil((config.BASE / config.NUM_TRAIN) * total_data_num).astype(int)
        add = np.ceil((config.BUDGET / config.NUM_TRAIN) * total_data_num).astype(int)
        print('Base number:', base)
        print('Budget each round:', add)
        
        data_num = []
        data_ratio_list = []
        loss_weight_list = []
        
        trainer = FederatedTrainer(device, config, logger)
        
        
        # Prepare initial data pools for each client
        for c in range(config.CLIENTS):
            # Setup deterministic data selection
            client_worker_init_fn = get_seed_worker(trial_seed + c * 100)
            g_labeled = torch.Generator()
            g_labeled.manual_seed(trial_seed + c * 100 + 10000)
            g_unlabeled = torch.Generator()
            g_unlabeled.manual_seed(trial_seed + c * 100 + 20000)
            
            data_list.append(data_splits[c])
        
            # Deterministic shuffling for initial split
            init_sample_rng = np.random.RandomState(trial_seed + c * 100)
            shuffled_indices = np.arange(len(data_splits[c]))
            init_sample_rng.shuffle(shuffled_indices)
            data_list[c] = [data_splits[c][i] for i in shuffled_indices]
            
            # Select initial labeled set
            labeled_set_list.append(data_list[c][:base[c]])
            
            # Compute class distribution for client dataset
            values, counts = np.unique(id2lab[np.array(data_list[c])], return_counts=True)
            dictionary = dict(zip(values, counts))
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            ratio /= np.sum(counts)
            data_ratio_list.append(ratio)
            
            # Compute class distribution for labeled set
            values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            loss_weight_list.append(torch.tensor(ratio, dtype=torch.float32).to(device))
            
            # Track samples per client
            data_num.append(len(labeled_set_list[c]))
            unlabeled_set_list.append(data_list[c][base[c]:])
            
            # Create data loaders
            private_train_loaders.append(DataLoader(
                cifar10_train, 
                batch_size=config.BATCH,
                sampler=SubsetRandomSampler(labeled_set_list[c]),
                num_workers=0,
                worker_init_fn=client_worker_init_fn,
                generator=g_labeled,
                pin_memory=True
            ))
            
            private_unlab_loaders.append(DataLoader(
                cifar10_train, 
                batch_size=config.BATCH,
                sampler=SubsetRandomSampler(unlabeled_set_list[c]),
                num_workers=0,
                worker_init_fn=client_worker_init_fn,
                generator=g_unlabeled,
                pin_memory=True
            ))
            
            # Initialize client models
            client_models.append(copy.deepcopy(base_model).to(device))
    
        data_num = np.array(data_num)

        # Log initial data distributions
        for c in range(config.CLIENTS):
            initial_class_labels = [id2lab[idx] for idx in labeled_set_list[c]]
            logger.log_sample_classes(0, initial_class_labels, c)
        
        # Update loss_weight_list in strategy_params and initialize strategy manager
        strategy_params['loss_weight_list'] = loss_weight_list
        strategy_manager = StrategyManager(**strategy_params)
        
        # If using strategies that need labeled set list or total clients
        if config.ACTIVE_LEARNING_STRATEGY in ["CoreSet", "AHFAL"]:
            strategy_manager.set_total_clients(config.CLIENTS)
            strategy_manager.set_labeled_set_list(labeled_set_list)

        # Create test loader
        test_loader = create_test_loader(cifar10_test, trial_seed)
        
        test_indices = list(range(len(cifar10_test)))
        test_rng = np.random.RandomState(trial_seed + 500)
        test_rng.shuffle(test_indices)
        val_size = int(0.2 * len(test_indices))
        val_indices = test_indices[:val_size]
        val_loader = create_val_loader(cifar10_test, trial_seed, val_indices)
        
        # Create dataloaders dictionary
        dataloaders = {
            'train-private': private_train_loaders,
            'unlab-private': private_unlab_loaders,
            'test': test_loader,
            'val': val_loader
        }
        
        # Initialize federated trainer
        trainer = FederatedTrainer(device, config, logger)
        trainer.set_loss_weights(loss_weight_list)
        trainer.set_data_num(data_num)
        
        for c in range(config.CLIENTS):
            trainer.update_client_distribution(c, labeled_set_list[c], cifar10_train)
        
        # Analyze variance across clients
        variance_stats = trainer.analyze_class_distribution_variance()
        
        # Calculate global class distribution
        global_distribution = trainer.aggregate_class_distributions()
        
        # Check if strategy needs global distribution
        if config.ACTIVE_LEARNING_STRATEGY in ["AHFAL"] and global_distribution is None:
            raise ValueError("Error: AHFAL strategy requires global class distribution, but none was computed. "
                            "Make sure there are labeled samples available on all clients.")
        
        # Variables to store the final model and optimizer states
        final_models = None
        final_optimizers = None
        final_schedulers = None
        
        # Active learning cycles
        for cycle in range(config.CYCLES):
            # Create server model
            if config.DATASET == "MNIST":
                server = resnet_mnist.preact_resnet8_mnist(num_classes=num_classes).to(device)
            else:
                server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
            models = {'clients': client_models, 'server': server}
            
            # Initialize criterion, optimizers, and schedulers
            criterion = nn.CrossEntropyLoss(reduction='none')
            
            optim_clients = []
            sched_clients = []
            
            for c in range(config.CLIENTS):
                optim_clients.append(optim.SGD(
                    models['clients'][c].parameters(), 
                    lr=config.LR,
                    momentum=config.MOMENTUM, 
                    weight_decay=config.WDECAY
                ))
                sched_clients.append(lr_scheduler.MultiStepLR(optim_clients[c], milestones=config.MILESTONES))
            
            optim_server = optim.SGD(
                models['server'].parameters(),
                lr=config.LR,
                momentum=config.MOMENTUM,
                weight_decay=config.WDECAY
            )
            
            sched_server = lr_scheduler.MultiStepLR(optim_server, milestones=config.MILESTONES)
            
            optimizers = {'clients': optim_clients, 'server': optim_server}
            schedulers = {'clients': sched_clients, 'server': sched_server}
            
            
            
            # Update client distribution in trainer
            for c in range(config.CLIENTS):
                trainer.update_client_distribution(c, labeled_set_list[c], cifar10_train)
            
            # Recalculate variance statistics for next cycle
            variance_stats = trainer.analyze_class_distribution_variance()
               
            # Count total labeled samples
            total_labels = sum(len(labeled_set_list[c]) for c in range(config.CLIENTS))
            
            # Prepare for next cycle
            private_train_loaders = []
            data_num = []
            loss_weight_list_2 = []
            server_state_dict = models['server'].state_dict()
            
            # Sample for annotations
            for c in range(config.CLIENTS):
                # Setup deterministic sampling
                cycle_worker_init_fn = get_seed_worker(trial_seed + c * 100 + cycle * 1000)
                g_labeled_cycle = torch.Generator()
                g_labeled_cycle.manual_seed(trial_seed + c * 100 + cycle * 1000 + 10000)
                g_unlabeled_cycle = torch.Generator()
                g_unlabeled_cycle.manual_seed(trial_seed + c * 100 + cycle * 1000 + 20000)
                
                # Shuffle unlabeled indices deterministically
                c_rng = np.random.RandomState(trial_seed + c * 500 + cycle * 50)
                unlabeled_indices = np.array(unlabeled_set_list[c])
                c_rng.shuffle(unlabeled_indices)
                unlabeled_set_list[c] = unlabeled_indices.tolist()
                
                # Create unlabeled loader for sequential sampling
                unlabeled_loader = DataLoader(
                    cifar10_select,
                    batch_size=config.BATCH,
                    sampler=SubsetSequentialSampler(unlabeled_set_list[c]),
                    num_workers=0,
                    worker_init_fn=cycle_worker_init_fn,
                    generator=g_unlabeled_cycle,
                    pin_memory=True
                )
                
                # Select samples using strategy manager
                if config.ACTIVE_LEARNING_STRATEGY in ["AHFAL"]:
                    # Pass global distribution when using AHFAL
                    selected_samples, remaining_unlabeled = strategy_manager.select_samples(
                        models['clients'][c],
                        models['server'],
                        unlabeled_loader,
                        c,
                        unlabeled_set_list[c],
                        add[c],
                        labeled_set=labeled_set_list[c],
                        seed=trial_seed + c * 100 + cycle * 1000,
                        global_class_distribution=global_distribution,  
                        class_variance_stats=variance_stats,  
                        current_round=cycle,                 
                        total_rounds=config.CYCLES           
                    )
                else:
                    selected_samples, remaining_unlabeled = strategy_manager.select_samples(
                        models['clients'][c],
                        models['server'],
                        unlabeled_loader,
                        c,
                        unlabeled_set_list[c],
                        add[c],
                        labeled_set=labeled_set_list[c],
                        seed=trial_seed + c * 100 + cycle * 1000
                    )
                
                # Log selected samples and their classes
                logger.log_selected_samples(cycle + 1, selected_samples, c)
                selected_classes = [id2lab[idx] for idx in selected_samples]
                logger.log_sample_classes(cycle + 1, selected_classes, c)
                
                # Update labeled and unlabeled sets
                labeled_set_list[c].extend(selected_samples)
                unlabeled_set_list[c] = remaining_unlabeled
                
                # Update class distribution
                values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
                ratio = np.zeros(num_classes)
                ratio[values] = counts
                
                # Update loss weights
                loss_weight_list_2.append(torch.tensor(ratio).to(device).float())
                data_num.append(len(labeled_set_list[c]))
                
                # Create updated dataloaders
                private_train_loaders.append(DataLoader(
                    cifar10_train,
                    batch_size=config.BATCH,
                    sampler=SubsetRandomSampler(labeled_set_list[c]),
                    num_workers=0,
                    worker_init_fn=cycle_worker_init_fn,
                    generator=g_labeled_cycle,
                    pin_memory=True
                ))
                
                private_unlab_loaders.append(DataLoader(
                    cifar10_train,
                    batch_size=config.BATCH,
                    sampler=SubsetRandomSampler(unlabeled_set_list[c]),
                    num_workers=0,
                    worker_init_fn=cycle_worker_init_fn,
                    generator=g_unlabeled_cycle,
                    pin_memory=True
                ))
            
            print(f"\n===== Starting Training for Cycle {cycle + 1} =====")
            trainer.train(
                    models, criterion, optimizers, schedulers, dataloaders, config.EPOCH, trial_seed
                )
            print(f"Training completed for cycle {cycle + 1}")
            
            # Evaluate server model
            acc_server = evaluate_model(models['server'], dataloaders['test'], device)
            logger.log_global_accuracy(cycle, acc_server)
            
            print('Trial {}/{} || Cycle {}/{} || Labelled sets size {}: server acc {:.2f}%'.format(
                trial + 1, config.TRIALS, cycle + 1, config.CYCLES, total_labels, acc_server))
            
            # Log the accumulated labeled samples for each client
            print("\n===== Accumulated Labeled Samples =====")
            for c in range(config.CLIENTS):
                print(f"Client {c}: {len(labeled_set_list[c])} samples")
            print(f"Total labeled samples: {sum(len(labeled_set_list[c]) for c in range(config.CLIENTS))}")
            print("=======================================\n")
            
            
            # Log per-class accuracies
            class_accuracies = evaluate_per_class_accuracy(models['server'], dataloaders['test'], device)
            logger.log_class_accuracies(cycle, class_accuracies)
            
            # Update for next cycle
            loss_weight_list = loss_weight_list_2
            dataloaders['train-private'] = private_train_loaders
            dataloaders['unlab-private'] = private_unlab_loaders
            data_num = np.array(data_num)
            trainer.set_loss_weights(loss_weight_list)
            trainer.set_data_num(data_num)
            
            # Record accuracy
            accuracies[trial].append(acc_server)
            
            # Save checkpoint after each cycle if enabled (by default)
            if args.save_checkpoints and (cycle + 1) % args.checkpoint_frequency == 0:
                print(f"\n===== Saving checkpoint after cycle {cycle + 1} =====")
                checkpoint_path = trainer.save_checkpoint(
                    models, optimizers, schedulers, cycle, config.COMMUNICATION,
                    labeled_set_list, unlabeled_set_list, data_num
                )
                print(f"Checkpoint saved to: {checkpoint_path}")
            
            # Store the final model and optimizer states for the final checkpoint
            final_models = models
            final_optimizers = optimizers
            final_schedulers = schedulers
            
            # Save logs
            logger.save_data()
        
        print('Accuracies for trial {}:'.format(trial), accuracies[trial])

        # Save checkpoint at the end of the training
        if args.save_checkpoints and final_models is not None:
            print(f"\n===== Saving final checkpoint for trial {trial+1} =====")
            checkpoint_path = trainer.save_checkpoint(
                final_models, final_optimizers, final_schedulers, 
                config.CYCLES-1, config.COMMUNICATION,
                labeled_set_list, unlabeled_set_list, data_num
            )
            print(f"Final checkpoint saved to: {checkpoint_path}")
            
            # Also save the accuracies to a file in the same directory
            accuracies_path = os.path.join(os.path.dirname(checkpoint_path), 'accuracies.npy')
            np.save(accuracies_path, np.array(accuracies, dtype=object))
            print(f"Accuracies saved to: {accuracies_path}")
            
    
    # Print overall results
    print('Accuracies by trial:')
    for trial_idx, trial_accs in enumerate(accuracies):
        if len(trial_accs) > 0:
            print(f"  Trial {trial_idx+1}: {trial_accs}, mean: {np.mean(trial_accs):.2f}%")
        else:
            print(f"  Trial {trial_idx+1}: No accuracy data recorded")
    
    # Calculate overall mean only for trials with data
    valid_means = []
    for trial_accs in accuracies:
        if len(trial_accs) > 0:
            valid_means.append(np.mean(trial_accs))
    
    if valid_means:
        print(f"Overall mean accuracy: {np.mean(valid_means):.2f}%")
    else:
        print("No accuracy data to calculate mean")
    

if __name__ == '__main__':
    main()