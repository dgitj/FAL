"""
Main entry point for federated active learning experiments.
Implements federated learning with various active learning sampling strategies.
"""

import argparse
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Import models
import models.preact_resnet as resnet
import models.preact_resnet_mnist as resnet_mnist
import models.mobilenet_v2 as mobilenet
import models.mobilenet_v2_mnist as mobilenet_mnist

# Import data utilities
from data.dirichlet_partitioner import dirichlet_balanced_partition
from data.sampler import SubsetSequentialSampler

# Import training module components
from training.trainer import FederatedTrainer
from training.evaluation import evaluate_model, evaluate_per_class_accuracy
from training.utils import (
    set_all_seeds, get_seed_worker, log_config, get_device
)
from query_strategies.strategy_manager import StrategyManager
from analysis.logger_strategy import FederatedALLogger
import config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Active Learning')
    parser.add_argument('--strategy', type=str, help='Active learning strategy to use')
    parser.add_argument('--confidence', type=float, default=0.0, help='Confidence threshold for PseudoEntropy (default: 0.0)')
    parser.add_argument('--cycles', type=int, help='Number of active learning cycles')
    parser.add_argument('--clients', type=int, help='Number of federated clients')
    parser.add_argument('--alpha', type=float, help='Dirichlet partition non-IID level')
    parser.add_argument('--budget', type=int, help='Active learning budget per cycle')
    parser.add_argument('--base', type=int, help='Initial labeled set size')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'SVHN', 'CIFAR100', 'MNIST'], help='Dataset to use')
    parser.add_argument('--model', type=str, choices=['resnet8', 'mobilenet_v2'], help='Model architecture to use')
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
        
        # Load datasets
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
        
        # Load datasets
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
        
        # Load datasets
        train_dataset = CIFAR100(dataset_dir, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(dataset_dir, train=False, download=True, transform=test_transform)
        select_dataset = CIFAR100(dataset_dir, train=True, download=True, transform=test_transform)
        
    elif dataset_name == "MNIST":
        train_transform = T.Compose([
            T.RandomCrop(size=28, padding=2),  # MNIST is 28x28
            T.ToTensor(),
            T.Normalize([0.1307], [0.3081])  # MNIST mean and std
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.1307], [0.3081])
        ])

        dataset_dir = config.DATA_ROOT
        
        # Load datasets
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

def main():
    """Main function to run federated active learning experiments."""
    # Parse arguments
    args = parse_arguments() 
    
    # Override configuration with command line arguments
    if args.strategy:
        config.ACTIVE_LEARNING_STRATEGY = args.strategy
        print(f"Using strategy: {config.ACTIVE_LEARNING_STRATEGY}")
    
    # Add dataset selection
    if args.dataset:
        config.DATASET = args.dataset
        # Update DATA_ROOT based on dataset
        if config.DATASET == "CIFAR10":
            config.DATA_ROOT = 'data/cifar-10-batches-py'
            config.NUM_CLASSES = 10
            config.NUM_TRAIN = 50000
        elif config.DATASET == "SVHN":
            config.DATA_ROOT = 'data/svhn'
            config.NUM_CLASSES = 10
            config.NUM_TRAIN = 73257  # SVHN train set size
        elif config.DATASET == "CIFAR100":
            config.DATA_ROOT = 'data/cifar-100-python'
            config.NUM_CLASSES = 100
            config.NUM_TRAIN = 50000
        elif config.DATASET == "MNIST":
            config.DATA_ROOT = 'data/mnist'
            config.NUM_CLASSES = 10
            config.NUM_TRAIN = 60000  # MNIST train set size
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
    
    if args.model:
        config.MODEL_ARCHITECTURE = args.model
        print(f"Using model architecture: {config.MODEL_ARCHITECTURE}")
    
    # Log configuration
    log_config(config)
    
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load datasets
    cifar10_train, cifar10_test, cifar10_select = load_datasets()
    
    # Prepare for trials
    accuracies = [[] for _ in range(config.TRIALS)]
    
    # Extract all labels
    indices = list(range(len(cifar10_train)))
    if config.DATASET == "CIFAR10":
        id2lab = [cifar10_train[id][1] for id in indices]
    elif config.DATASET == "SVHN":
        # For SVHN, we need to use the labels attribute directly
        id2lab = [cifar10_train.labels[id] for id in indices]
    elif config.DATASET == "MNIST":
        id2lab = [cifar10_train[id][1] for id in indices]
    else:
        id2lab = [cifar10_train[id][1] for id in indices]  # Default to CIFAR10 format
    id2lab = np.array(id2lab)
    
    # Run trials
    for trial in range(config.TRIALS):
        trial_seed = config.SEED + config.TRIAL_SEED_OFFSET * (trial + 1)
        set_all_seeds(trial_seed)
        
        print(f"\n=== Trial {trial+1}/{config.TRIALS} (Seed: {trial_seed}) ===\n")
        print(f"Generating Dirichlet partition with alpha {config.ALPHA}, seed {trial_seed} for {config.CLIENTS} clients...")
        
        # Create non-IID data partitioning
        data_splits = dirichlet_balanced_partition(cifar10_train, config.CLIENTS, alpha=config.ALPHA, seed=trial_seed)
        
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
            'loss_weight_list': [],  # Will be populated during initialization or resume
            'device': device
        }
        
        # Add confidence threshold for AHFAL if specified
        if config.ACTIVE_LEARNING_STRATEGY in ["AHFAL"] and args.confidence is not None:
            strategy_params['confidence_threshold'] = args.confidence
            print(f"Setting AHFAL confidence threshold to: {args.confidence}")
        
        # Prepare client data
        # Initialize model based on architecture setting
        if config.MODEL_ARCHITECTURE == "resnet8":
            if config.DATASET == "MNIST":
                base_model = resnet_mnist.preact_resnet8_mnist(num_classes=num_classes)
            else:
                base_model = resnet.preact_resnet8_cifar(num_classes=num_classes)
        elif config.MODEL_ARCHITECTURE == "mobilenet_v2":
            if config.DATASET == "MNIST":
                base_model = mobilenet_mnist.mobilenet_v2_mnist(num_classes=num_classes)
            else:
                base_model = mobilenet.mobilenet_v2_cifar(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model architecture: {config.MODEL_ARCHITECTURE}")
        
        # For compatibility, keep the variable name as resnet8
        resnet8 = base_model
        client_models = []
        data_list = []
        total_data_num = [len(data_splits[c]) for c in range(config.CLIENTS)]
        total_data_num = np.array(total_data_num)
        
        # Calculate base and budget sizes
        # base = np.ceil((config.BASE / config.NUM_TRAIN) * total_data_num).astype(int)
        # add = np.ceil((config.BUDGET / config.NUM_TRAIN) * total_data_num).astype(int)
        
        base_value = int(np.ceil(config.BASE / config.CLIENTS))
        budget_value = int(np.ceil(config.BUDGET / config.CLIENTS))

        # Create arrays with these values for all clients
        base = np.full(config.CLIENTS, base_value)
        add = np.full(config.CLIENTS, budget_value)
        print('Base number:', base)
        print('Budget each round:', add)
        
        data_num = []
        data_ratio_list = []
        loss_weight_list = []
        
        # Initialize trainer with common settings
        trainer = FederatedTrainer(device, config, logger)
        
        # Track starting cycle for resuming from checkpoint
        cycle_start = 0
        
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
            client_models.append(copy.deepcopy(resnet8).to(device))
        
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
        
        # Create dataloaders dictionary
        dataloaders = {
            'train-private': private_train_loaders,
            'unlab-private': private_unlab_loaders,
            'test': test_loader
        }
        
        # Initialize federated trainer
        trainer = FederatedTrainer(device, config, logger)
        trainer.set_loss_weights(loss_weight_list)
        trainer.set_data_num(data_num)
        
        # Calculate initial class distributions and variance analysis (only once)
        print("\n===== Initial Class Distribution Analysis =====")
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
        
        
        # Active learning cycles
        for cycle in range(cycle_start, config.CYCLES):
            # Create server model
            if config.MODEL_ARCHITECTURE == "resnet8":
                if config.DATASET == "MNIST":
                    server = resnet_mnist.preact_resnet8_mnist(num_classes=num_classes).to(device)
                else:
                    server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
            elif config.MODEL_ARCHITECTURE == "mobilenet_v2":
                if config.DATASET == "MNIST":
                    server = mobilenet_mnist.mobilenet_v2_mnist(num_classes=num_classes).to(device)
                else:
                    server = mobilenet.mobilenet_v2_cifar(num_classes=num_classes).to(device)
            else:
                raise ValueError(f"Unknown model architecture: {config.MODEL_ARCHITECTURE}")
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
            
            trainer.train(
                models, criterion, optimizers, schedulers, dataloaders, config.EPOCH, trial_seed
            )
            
            # Update class distribution statistics after selection
            print("\n===== Updating Class Distribution Analysis =====\n")
            
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
                        global_class_distribution=global_distribution,  # Pass global distribution
                        class_variance_stats=variance_stats,  # Add variance stats
                        current_round=cycle,                 # Add current round
                        total_rounds=config.CYCLES           # Add total rounds
                    )
                else:
                    # Original call for other strategies
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
            # Save logs
            logger.save_data()
        
        print('Accuracies for trial {}:'.format(trial), accuracies[trial])
    
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