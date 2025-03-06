import os
import random
import copy
import json
import time
from datetime import timedelta
import importlib
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.distributions import Beta
import torch.multiprocessing as mp

# Set multiprocessing start method early
if __name__ == '__main__':
    # 'spawn' is more reliable with CUDA
    mp.set_start_method('spawn', force=True)


# Set FAL parameters uses parameters from config.py if not explicitly changed
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default=config.ACTIVE_LEARNING_STRATEGY)
parser.add_argument("--clients", type=int, default=config.CLIENTS)
parser.add_argument("--epochs", type=int, default=config.EPOCH)
parser.add_argument("--communication_rounds", type=int, default=config.COMMUNICATION)
args = parser.parse_args()

ACTIVE_LEARNING_STRATEGY = args.strategy
CLIENTS = args.clients
EPOCH = args.epochs
COMMUNICATION = args.communication_rounds



# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")
print(f"Number of CPU cores: {mp.cpu_count()}")

from query_strategies.strategy_manager import StrategyManager
import models.preact_resnet as resnet
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from data.sampler import SubsetSequentialSampler, SubsetSequentialRandomSampler

def format_time(seconds):
    """Format time in a human-readable way"""
    return str(timedelta(seconds=int(seconds)))

# Import the multiprocessing functions
from multiprocessing_utils import (
    parallel_train_clients, 
    parallel_select_samples
)

# Configuration
from config import *

# Load data
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

cifar10_train = CIFAR10(cifar10_dataset_dir, train=True, download=True, transform=cifar10_train_transform)
cifar10_test  = CIFAR10(cifar10_dataset_dir, train=False, download=True, transform=cifar10_test_transform)
cifar10_select = CIFAR10(cifar10_dataset_dir, train=True, download=True, transform=cifar10_test_transform)

def read_data(dataloader):
    while True:
        for data in dataloader:
            yield data

iters = 0

def test(models, dataloaders, mode='test'):
    models['server'].eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            scores, _ = models['server'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, num_processes=None):
    print('>> Training with multiprocessing')

    overall_start = time.time()
    
    for com in range(COMMUNICATION):
        comm_start = time.time()
        print(f"\n--- Starting Communication Round {com + 1}/{COMMUNICATION} ---")
        # Select clients 
        if com < COMMUNICATION-1:
            selected_clients_id = np.random.choice(CLIENTS, int(CLIENTS * RATIO), replace=False)
        else:
            selected_clients_id = range(CLIENTS)

        if num_processes is None:
            import multiprocessing as mp
            actual_processes = min(len(selected_clients_id), mp.cpu_count())
        else:
            actual_processes = min(len(selected_clients_id), num_processes)
            
        print(f"Training {len(selected_clients_id)} clients in parallel with {actual_processes} processes")


        server_state_dict = models['server'].state_dict()

        # Broadcast server model to selected clients
        for c in selected_clients_id:
            models['clients'][c].load_state_dict(server_state_dict, strict=False)

        # Local updates in parallel
        client_start = time.time()
        
        results = parallel_train_clients(
            models, 
            criterion, 
            optimizers, 
            schedulers, 
            dataloaders, 
            selected_clients_id, 
            num_epochs,
            num_processes
        )
        
        # Update client models with trained states
        for result in results:
            client_id = result['client_id']
            models['clients'][client_id].load_state_dict(result['state_dict'])
            # schedulers['clients'][client_id].step()
            
        client_end = time.time()
        client_time = client_end - client_start
        print(f"Client training completed in {format_time(client_time)} ({client_time:.2f}s)")

        # Aggregation
        agg_start = time.time()
        print("Aggregating model updates...")

        local_states = [
            copy.deepcopy(models['clients'][c].state_dict())
            for c in selected_clients_id
        ]

        selected_data_num = data_num[selected_clients_id]
        model_state = local_states[0]

        for key in local_states[0]:
            model_state[key] = model_state[key] * selected_data_num[0]
            for i in range(1, len(selected_clients_id)):
                model_state[key] = model_state[key].float() + local_states[i][key].float() * selected_data_num[i]
            model_state[key] = model_state[key].float() / np.sum(selected_data_num)
            
        models['server'].load_state_dict(model_state, strict=False)
        agg_end = time.time()
        agg_time = agg_end - agg_start
        print(f"Model aggregation completed in {format_time(agg_time)} ({agg_time:.2f}s)")

        comm_end = time.time()
        comm_time = comm_end - comm_start
        print(f"--- Communication Round {com + 1}/{COMMUNICATION} completed in {format_time(comm_time)} ({comm_time:.2f}s) ---")
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    print(f"\n>> Cycle completed in {format_time(overall_time)} ({overall_time:.2f}s)")



##
# Main
if __name__ == '__main__':
    # Set the number of processes to use for multiprocessing
    # You can adjust this based on your hardware capabilities
    NUM_PROCESSES = min(CLIENTS, mp.cpu_count())

    accuracies = [[] for _ in range(TRIALS)]

    indices = list(range(NUM_TRAIN))
    id2lab = []
    for id in indices:
        id2lab.append(cifar10_train[id][1])
    id2lab = np.array(id2lab)

    with open(f"distribution/alpha0.1_cifar10_{CLIENTS}clients_var0.1_seed42.json") as json_file:
        data_splits = json.load(json_file)
        
    for trial in range(TRIALS):
        random.seed(100 + trial)
        labeled_set_list = []
        unlabeled_set_list = []
        pseudo_set = []

        private_train_loaders = []
        private_unlab_loaders = []
        num_classes = 10 # CIFAR10

        rest_data = indices.copy()

        print('Query Strategy:', ACTIVE_LEARNING_STRATEGY)
        print('Number of clients:', CLIENTS)
        print('Number of epochs:', EPOCH)
        print('Number of communication rounds:', COMMUNICATION)

        num_per_client = int(len(rest_data) / CLIENTS)
        client_models = []
        data_list = []
        total_data_num = []
        
        for c in range(CLIENTS):
            total_data_num.append(len(data_splits[c]))
        
        total_data_num = np.array(total_data_num)
        base = np.ceil((BASE / NUM_TRAIN)*total_data_num).astype(int)
        add = np.ceil((BUDGET / NUM_TRAIN) * total_data_num).astype(int)
        
        print('base number: ',base)
        print('budget each round: ', add)
        
        data_num = []
        data_ratio_list = []
        loss_weight_list = []

        # prepare initial data pools
        for c in range(CLIENTS):
            data_list.append(data_splits[c])
            random.shuffle(data_list[c])
            labeled_set_list.append(data_list[c][:base[c]])
            
            values, counts = np.unique(id2lab[np.array(data_list[c])], return_counts=True)
            dictionary = dict(zip(values, counts))
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            ratio /= np.sum(counts)
            data_ratio_list.append(ratio)
            print(f'client {c}, ratio {ratio}')

            values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            loss_weight_list.append(torch.tensor(ratio, dtype=torch.float32).to(device))

            data_num.append(len(labeled_set_list[c]))
            unlabeled_set_list.append(data_list[c][base[c]:])
            
            private_train_loaders.append(DataLoader(
                cifar10_train, 
                batch_size=BATCH,
                sampler=SubsetSequentialRandomSampler(labeled_set_list[c]),
                num_workers=2,
                pin_memory=True
            ))
            
            private_unlab_loaders.append(DataLoader(
                cifar10_train, 
                batch_size=BATCH,
                sampler=SubsetSequentialRandomSampler(unlabeled_set_list[c]),
                num_workers=2,
                pin_memory=True
            ))

            client_models.append(copy.deepcopy(resnet.preact_resnet8_cifar(num_classes=num_classes)).to(device))
            
        data_num = np.array(data_num)

        # Initialize strategy manager
        strategy_manager = StrategyManager(
            strategy_name=ACTIVE_LEARNING_STRATEGY, 
            loss_weight_list=loss_weight_list,
            device=device
        )

        test_loader = DataLoader(cifar10_test, batch_size=BATCH)
        dataloaders = {'train-private': private_train_loaders, 'unlab-private': private_unlab_loaders, 'test': test_loader}
        
        # Model
        models = {'clients': client_models, 'server': None}
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            cycle_start = time.time()
            print(f"\n===== Starting Active Learning Cycle {cycle+1}/{CYCLES} =====")
            
            server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
            models['server'] = server
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
                
                sched_clients.append(lr_scheduler.MultiStepLR(
                    optim_clients[c], 
                    milestones=MILESTONES
                ))

            optim_server = optim.SGD(
                models['server'].parameters(), 
                lr=LR,
                momentum=MOMENTUM, 
                weight_decay=WDECAY
            )

            sched_server = lr_scheduler.MultiStepLR(
                optim_server, 
                milestones=MILESTONES
            )

            optimizers = {'clients': optim_clients, 'server': optim_server}
            schedulers = {'clients': sched_clients, 'server': sched_server}

            # Train with multiprocessing
            train(
                models, 
                criterion, 
                optimizers, 
                schedulers, 
                dataloaders, 
                EPOCH, 
                NUM_PROCESSES
            )
            
             # Evaluation
            eval_start = time.time()
            acc_server = test(models, dataloaders, mode='test')
            eval_end = time.time()
            eval_time = eval_end - eval_start

            # Calculate total labels
            total_labels = sum(len(labeled_set) for labeled_set in labeled_set_list)

            print(f'Trial {trial + 1}/{TRIALS} || Cycle {cycle + 1}/{CYCLES} || Labelled sets size {total_labels}: server acc {acc_server}')
            print(f'Evaluation completed in {format_time(eval_time)} ({eval_time:.2f}s)')

            # Prepare for next cycle
            private_train_loaders = []
            data_num = []
            loss_weight_list_2 = []
            server_state_dict = models['server'].state_dict()

            # Perform parallel sample selection
            print(f"Selecting samples in parallel with {NUM_PROCESSES} processes")
            start_time = time.time()
            
            selection_results = parallel_select_samples(
                models,
                strategy_manager,
                unlabeled_set_list,
                add,
                NUM_PROCESSES
            )
            
            end_time = time.time()
            print(f"Sample selection completed in {end_time - start_time:.2f} seconds")

            # Process the results
            for result in selection_results:
                c = result['client_id']
                selected_samples = result['selected_samples']
                remaining_unlabeled = result['remaining_unlabeled']
                
                # Update model with server state
                models['clients'][c].load_state_dict(server_state_dict, strict=False)
                
                # Update labeled and unlabeled sets
                labeled_set_list[c].extend(selected_samples)
                unlabeled_set_list[c] = remaining_unlabeled

                # Compute new class distributions
                values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
                ratio = np.zeros(num_classes)
                ratio[values] = counts
                loss_weight_list_2.append(torch.tensor(ratio).to(device).float())
                
                data_num.append(len(labeled_set_list[c]))

                # Update dataloaders with new data pools
                private_train_loaders.append(DataLoader(
                    cifar10_train, 
                    batch_size=BATCH,
                    sampler=SubsetSequentialRandomSampler(labeled_set_list[c]),
                    num_workers=2,
                    pin_memory=True
                ))
                
                private_unlab_loaders.append(DataLoader(
                    cifar10_train, 
                    batch_size=BATCH,
                    sampler=SubsetSequentialRandomSampler(unlabeled_set_list[c]),
                    num_workers=2,
                    pin_memory=True
                ))

            # Evaluation
            acc_server = test(models, dataloaders, mode='test')
            print(f'Trial {trial+1}/{TRIALS} || Cycle {cycle+1}/{CYCLES} || Labeled sets size {total_labels}: server acc {acc_server}')

            # Update distributions
            loss_weight_list = loss_weight_list_2
            dataloaders['train-private'] = private_train_loaders
            dataloaders['unlab-private'] = private_unlab_loaders
            data_num = np.array(data_num)
            accuracies[trial].append(acc_server)
            
        print(f'Accuracies for trial {trial}:', accuracies[trial])

    print('Accuracies means:', np.array(accuracies).mean(1))