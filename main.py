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
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Import model
import models.preact_resnet as resnet

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
from config import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Federated Active Learning')
    parser.add_argument('--strategy', type=str, help='Active learning strategy to use')
    return parser.parse_args()


def load_datasets():
    """Load and prepare CIFAR10 datasets."""
    # CIFAR10 transformations
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T

    cifar10_train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    cifar10_test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    cifar10_dataset_dir = DATA_ROOT

    # Load datasets
    cifar10_train = CIFAR10(cifar10_dataset_dir, train=True, download=True, transform=cifar10_train_transform)
    cifar10_test = CIFAR10(cifar10_dataset_dir, train=False, download=True, transform=cifar10_test_transform)
    cifar10_select = CIFAR10(cifar10_dataset_dir, train=True, download=True, transform=cifar10_test_transform)

    return cifar10_train, cifar10_test, cifar10_select


def create_test_loader(dataset, trial_seed, batch_size=BATCH):
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
    if args.strategy:
        config.ACTIVE_LEARNING_STRATEGY = args.strategy
    
    # Log configuration
    log_config(config)
    
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = create_results_dir("results")
    
    # Load datasets
    cifar10_train, cifar10_test, cifar10_select = load_datasets()
    
    # Prepare for trials
    accuracies = [[] for _ in range(TRIALS)]
    
    # Extract all labels
    indices = list(range(NUM_TRAIN))
    id2lab = [cifar10_train[id][1] for id in indices]
    id2lab = np.array(id2lab)
    
    # Run trials
    for trial in range(TRIALS):
        trial_seed = SEED + TRIAL_SEED_OFFSET * (trial + 1)
        set_all_seeds(trial_seed)
        
        print(f"\n=== Trial {trial+1}/{TRIALS} (Seed: {trial_seed}) ===\n")
        print(f"Generating Dirichlet partition with alpha {ALPHA}, seed {trial_seed} for {CLIENTS} clients...")
        
        # Create non-IID data partitioning
        data_splits = dirichlet_balanced_partition(cifar10_train, CLIENTS, alpha=ALPHA, seed=trial_seed)
        
        # Initialize logger
        logger = FederatedALLogger(
            strategy_name=ACTIVE_LEARNING_STRATEGY,
            num_clients=CLIENTS,
            num_classes=10,
            trial_id=trial+1 
        )
        
        # Setup for clients
        labeled_set_list = []
        unlabeled_set_list = []
        private_train_loaders = []
        private_unlab_loaders = []
        num_classes = 10
        
        print('Query Strategy:', ACTIVE_LEARNING_STRATEGY)
        print('Number of clients:', CLIENTS)
        print('Number of epochs:', EPOCH)
        print('Number of communication rounds:', COMMUNICATION)
        
        # Prepare client data
        resnet8 = resnet.preact_resnet8_cifar(num_classes=num_classes)
        client_models = []
        data_list = []
        total_data_num = [len(data_splits[c]) for c in range(CLIENTS)]
        total_data_num = np.array(total_data_num)
        
        # Calculate base and budget sizes
        base = np.ceil((BASE / NUM_TRAIN) * total_data_num).astype(int)
        add = np.ceil((BUDGET / NUM_TRAIN) * total_data_num).astype(int)
        print('Base number:', base)
        print('Budget each round:', add)
        
        data_num = []
        data_ratio_list = []
        loss_weight_list = []
        
        # Prepare initial data pools for each client
        for c in range(CLIENTS):
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
                batch_size=BATCH,
                sampler=SubsetRandomSampler(labeled_set_list[c]),
                num_workers=0,
                worker_init_fn=client_worker_init_fn,
                generator=g_labeled,
                pin_memory=True
            ))
            
            private_unlab_loaders.append(DataLoader(
                cifar10_train, 
                batch_size=BATCH,
                sampler=SubsetRandomSampler(unlabeled_set_list[c]),
                num_workers=0,
                worker_init_fn=client_worker_init_fn,
                generator=g_unlabeled,
                pin_memory=True
            ))
            
            # Initialize client models
            #client_models.append(resnet.preact_resnet8_cifar(num_classes=num_classes).to(device))
            client_models.append(copy.deepcopy(resnet8).to(device))
        
        data_num = np.array(data_num)
        
        # Log initial data distributions
        for c in range(CLIENTS):
            initial_class_labels = [id2lab[idx] for idx in labeled_set_list[c]]
            logger.log_sample_classes(0, initial_class_labels, c)
        
        # Initialize strategy manager
        strategy_manager = StrategyManager(
            strategy_name=ACTIVE_LEARNING_STRATEGY,
            loss_weight_list=loss_weight_list,
            device=device
        )
        # If using GlobalOptimal strategy, set the total number of clients
        if ACTIVE_LEARNING_STRATEGY == "GlobalOptimal":
            strategy_manager.set_total_clients(CLIENTS)
        
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
        
        # Active learning cycles
        for cycle in range(CYCLES):
            # Create server model
            server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
            models = {'clients': client_models, 'server': server}
            
            # Initialize criterion, optimizers, and schedulers
            criterion = nn.CrossEntropyLoss(reduction='none')
            
            optim_clients = []
            sched_clients = []
            
            for c in range(CLIENTS):
                optim_clients.append(optim.SGD(
                    models['clients'][c].parameters(), 
                    lr=LR,
                    momentum=MOMENTUM, 
                    weight_decay=WDECAY
                ))
                sched_clients.append(lr_scheduler.MultiStepLR(optim_clients[c], milestones=MILESTONES))
            
            optim_server = optim.SGD(
                models['server'].parameters(),
                lr=LR,
                momentum=MOMENTUM,
                weight_decay=WDECAY
            )
            
            sched_server = lr_scheduler.MultiStepLR(optim_server, milestones=MILESTONES)
            
            optimizers = {'clients': optim_clients, 'server': optim_server}
            schedulers = {'clients': sched_clients, 'server': sched_server}
            
            # Train
            trainer.train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, trial_seed)
            
            # Count total labeled samples
            total_labels = sum(len(labeled_set_list[c]) for c in range(CLIENTS))
            
            # Prepare for next cycle
            private_train_loaders = []
            data_num = []
            loss_weight_list_2 = []
            server_state_dict = models['server'].state_dict()
            
            # Log model distances at beginning of cycle
            if cycle == 0:
                model_distances = {}
                for c in range(CLIENTS):
                    distance = logger.calculate_model_distance(models['clients'][c], models['server'])
                    model_distances[c] = distance
                
                logger.log_model_distances(cycle, model_distances)
            
            # Sample for annotations
            for c in range(CLIENTS):
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
                    batch_size=BATCH,
                    sampler=SubsetSequentialSampler(unlabeled_set_list[c]),
                    num_workers=0,
                    worker_init_fn=cycle_worker_init_fn,
                    generator=g_unlabeled_cycle,
                    pin_memory=True
                )
                
                # Select samples using strategy manager
                selected_samples, remaining_unlabeled = strategy_manager.select_samples(
                    models['clients'][c],
                    models['server'],
                    unlabeled_loader,
                    c,
                    unlabeled_set_list[c],
                    add[c],
                    seed=trial_seed + c * 100 + cycle * 1000
                )
                
                # Load server model to client for next round
                models['clients'][c].load_state_dict(server_state_dict, strict=False)
                
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
                    batch_size=BATCH,
                    sampler=SubsetRandomSampler(labeled_set_list[c]),
                    num_workers=0,
                    worker_init_fn=cycle_worker_init_fn,
                    generator=g_labeled_cycle,
                    pin_memory=True
                ))
                
                private_unlab_loaders.append(DataLoader(
                    cifar10_train,
                    batch_size=BATCH,
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
                trial + 1, TRIALS, cycle + 1, CYCLES, total_labels, acc_server))
            
            # Log the accumulated labeled samples for each client
            print("\n===== Accumulated Labeled Samples =====")
            for c in range(CLIENTS):
                print(f"Client {c}: {len(labeled_set_list[c])} samples")
            print(f"Total labeled samples: {sum(len(labeled_set_list[c]) for c in range(CLIENTS))}")
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

        # Add this at the end of each trial (around line 380-390)
        if ACTIVE_LEARNING_STRATEGY == "GlobalOptimal":
            print("\n========== GLOBAL OPTIMAL STRATEGY SUMMARY ==========")
            print(f"Trial {trial+1}/{TRIALS}")
            
            # Calculate standard deviation and coefficient of variation of dataset sizes
            final_sizes = np.array([len(labeled_set_list[c]) for c in range(CLIENTS)])
            mean_size = np.mean(final_sizes)
            std_size = np.std(final_sizes)
            cv = std_size / mean_size * 100  # Coefficient of variation as percentage
            
            print(f"Mean labeled samples per client: {mean_size:.2f}")
            print(f"Standard deviation: {std_size:.2f}")
            print(f"Coefficient of variation: {cv:.2f}%")
            
            # Show min and max
            min_idx = np.argmin(final_sizes)
            max_idx = np.argmax(final_sizes)
            print(f"Client with min samples: Client {min_idx} ({final_sizes[min_idx]} samples)")
            print(f"Client with max samples: Client {max_idx} ({final_sizes[max_idx]} samples)")
            print(f"Imbalance ratio (max/min): {final_sizes[max_idx]/final_sizes[min_idx]:.2f}x")
            
            print("=======================================================\n")
        
    
    # Print overall results
    print('Accuracies means:', np.array(accuracies).mean(1))
    

if __name__ == '__main__':
    main()