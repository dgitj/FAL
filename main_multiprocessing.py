# Python
import os
import random
import copy
import json
import time
import importlib
import functools
import multiprocessing
from multiprocessing import Pool

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.distributions import Beta

# Dirichlet partitioner
from data.dirichlet_partitioner import dirichlet_balanced_partition

# Device
if torch.cuda.is_available():   
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

from query_strategies.strategy_manager import StrategyManager

# Model
import models.preact_resnet as resnet

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

# Utils
from config import *
from tqdm import tqdm

def set_all_seeds(seed):
    """Set all seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def seed_worker_fn(base_seed, worker_id):
    """Sets unique seed for each dataloader worker"""
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)

# Function to create a partial application (picklable)
def get_seed_worker(base_seed):
    """Creates a worker initialization function with the given base seed"""
    return functools.partial(seed_worker_fn, base_seed)

# dataset
from data.sampler import SubsetSequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

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

class BSMLoss(nn.Module):
    """
    balanced softmax loss
    """
    log_softmax = nn.LogSoftmax()

    def __init__(self):
        super().__init__()

    def log_softmax_weighted(self, x, target, loss_weights):
        c = x.max()
        diff = x - c
        target_weight = loss_weights[target]
        logsumexp = torch.log((loss_weights * torch.exp(diff)))
        return torch.log(target_weight).unsqueeze(0).repeat(x.size(0),1) - diff - logsumexp

    def forward(self, logits, target, loss_weights):
        log_probabilities = self.log_softmax_weighted(logits, target = target, loss_weights = loss_weights)
        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()

def read_data(dataloader):
    while True:
        for data in dataloader:
            yield data

iters = 0

# Function to train a single client (for parallel execution)
def train_client_worker(client_id, client_model_state, server_model_state, labeled_set, 
                        unlabeled_set, loss_weight_list, selected_clients_id, com, 
                        num_epochs, trial_seed, mode="Vanilla"):
    """Trains a single client and returns its updated model state dict"""
    try:
        # Set device (each worker should use its own device or CPU)
        worker_device = torch.device("cpu")  # Use CPU for parallel workers
        
        # Initialize client model with server state
        model = resnet.preact_resnet8_cifar(num_classes=10).to(worker_device)
        model.load_state_dict(server_model_state)
        model.train()
        
        # Initialize optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Create dataloaders
        client_seed = int(trial_seed + client_id * 1000 + com * 10)
        
        # Set client-specific seeds
        torch.manual_seed(client_seed)
        np.random.seed(client_seed)
        random.seed(client_seed)
        
        client_worker_init_fn = get_seed_worker(client_seed)
        
        # For labeled data loaders
        g_labeled = torch.Generator()
        g_labeled.manual_seed(client_seed + 10000)
        
        # For unlabeled data loaders
        g_unlabeled = torch.Generator()
        g_unlabeled.manual_seed(client_seed + 20000)
        
        train_loader = DataLoader(cifar10_train, batch_size=BATCH,
                           sampler=SubsetRandomSampler(labeled_set),
                           num_workers=0, worker_init_fn=client_worker_init_fn,
                           generator=g_labeled, pin_memory=False)
        
        if mode == "KCFU":
            unlab_loader = DataLoader(cifar10_train, batch_size=BATCH,
                                sampler=SubsetRandomSampler(unlabeled_set),
                                num_workers=0, worker_init_fn=client_worker_init_fn,
                                generator=g_unlabeled, pin_memory=False)
            unlab_set = read_data(unlab_loader)
        
        # Train for the specified number of epochs
        for epoch in range(num_epochs):
            # Set seed for this epoch
            batch_epoch_seed = client_seed + epoch
            torch.manual_seed(batch_epoch_seed)
            np.random.seed(batch_epoch_seed)
            random.seed(batch_epoch_seed)
            
            if mode == "Vanilla":
                # Train with vanilla method
                for batch_idx, data in enumerate(train_loader):
                    # Batch-specific seed
                    batch_seed = batch_epoch_seed + batch_idx
                    torch.manual_seed(batch_seed)
                    np.random.seed(batch_seed)
                    random.seed(batch_seed)
                    
                    # Move data to device
                    inputs, labels = data[0].to(worker_device), data[1].to(worker_device)
                    
                    # Forward pass
                    scores, _ = model(inputs)
                    loss = torch.sum(criterion(scores, labels)) / labels.size(0)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            elif mode == "KCFU":
                # KCFU training method
                kld = nn.KLDivLoss(reduce=False)
                server_model = resnet.preact_resnet8_cifar(num_classes=10).to(worker_device)
                server_model.load_state_dict(server_model_state)
                server_model.eval()
                
                for batch_idx, data in enumerate(train_loader):
                    # Batch-specific seed
                    batch_seed = batch_epoch_seed + batch_idx
                    torch.manual_seed(batch_seed)
                    np.random.seed(batch_seed)
                    random.seed(batch_seed)
                    
                    # Move data to device
                    inputs, labels = data[0].to(worker_device), data[1].to(worker_device)
                    
                    unlab_data = next(unlab_set)
                    unlab_inputs = unlab_data[0].to(worker_device)

                    # deterministic beta sampling
                    m = Beta(torch.FloatTensor([BETA[0]]).item(), torch.FloatTensor([BETA[1]]).item())
                    batch_rng = np.random.RandomState(batch_seed)
                    beta_samples = batch_rng.beta(BETA[0], BETA[1], size=unlab_inputs.size(0))
                    beta_0 = torch.FloatTensor(beta_samples).to(worker_device)
                    beta = beta_0.view(unlab_inputs.size(0), 1, 1, 1)

                    # deterministic index selection
                    indices = batch_rng.choice(unlab_inputs.size(0), size=unlab_inputs.size(0), replace=False)
                    
                    mixed_inputs = beta * unlab_inputs + (1 - beta) * unlab_inputs[indices,...]

                    scores, _ = model(inputs)

                    optimizer.zero_grad()

                    with torch.no_grad():
                        scores_unlab_t, _ = server_model(mixed_inputs)

                    scores_unlab, _ = model(mixed_inputs)
                    _, pred_labels = torch.max((scores_unlab_t.data), 1)

                    # find the \Gamma weight matrix
                    client_loss_weight = torch.tensor(loss_weight_list[client_id], dtype=torch.float32).to(worker_device)
                    mask = (client_loss_weight > 0).float()
                    weight_ratios = client_loss_weight.sum() / (client_loss_weight + 1e-6)
                    weight_ratios *= mask
                    weights = beta_0 * (weight_ratios / weight_ratios.sum())[pred_labels] + (1-beta_0)*(weight_ratios / weight_ratios.sum())[pred_labels[indices]]

                    # compensatory loss
                    distil_loss = int(com>0)*(weights*kld(F.log_softmax(scores_unlab, -1), F.softmax(scores_unlab_t.detach(), -1)).mean(1)).mean()

                    spc = client_loss_weight
                    spc = spc.unsqueeze(0).expand(labels.size(0), -1)
                    scores = scores + spc.log()

                    # client loss
                    loss = torch.sum(criterion(scores, labels)) / labels.size(0)

                    # KCFU loss
                    (loss + distil_loss).backward()
                    optimizer.step()
            
            # Step the scheduler
            scheduler.step()
        
        # Return the final state dict
        return model.state_dict()
    
    except Exception as e:
        print(f"Error in worker process for client {client_id}: {str(e)}")
        # Return empty state dict on error
        return {}

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

# Initialize a global pool
def init_worker():
    """Initialize worker process"""
    # Force CPU use for workers to avoid CUDA issues
    import torch
    torch.set_num_threads(1)  # Limit CPU threads per worker

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, trial_seed):
    print('>> Train a Model.')

    # Create process pool
    num_workers = min(multiprocessing.cpu_count(), CLIENTS)
    print(f"Using {num_workers} worker processes for parallel training")
    
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        for com in range(COMMUNICATION):
            # Deterministic client selection
            rng = np.random.RandomState(trial_seed + com * 100)

            if com < COMMUNICATION-1:
                selected_clients_id = rng.choice(CLIENTS, int(CLIENTS * RATIO), replace=False)
            else:
                selected_clients_id = range(CLIENTS)
                
            # Get server model state to broadcast to clients
            server_state_dict = models['server'].state_dict()
            
            # Prepare training tasks for parallel execution
            training_tasks = []
            for c in selected_clients_id:
                # Create a task for each client
                if LOCAL_MODEL_UPDATE == "Vanilla":
                    task = (c, models['clients'][c].state_dict(), server_state_dict, 
                            dataloaders['labeled_sets'][c], dataloaders['unlabeled_sets'][c],
                            loss_weight_list, selected_clients_id, com, num_epochs, trial_seed, "Vanilla")
                else:  # KCFU
                    task = (c, models['clients'][c].state_dict(), server_state_dict, 
                            dataloaders['labeled_sets'][c], dataloaders['unlabeled_sets'][c],
                            loss_weight_list, selected_clients_id, com, num_epochs, trial_seed, "KCFU")
                training_tasks.append(task)
            
            # Execute training in parallel
            start = time.time()
            print(f"Starting parallel training for {len(training_tasks)} clients...")
            results = pool.starmap(train_client_worker, training_tasks)
            end = time.time()
            print('Average time per epoch:', (end-start)/num_epochs)
            
            # Update client models with trained weights
            local_states = []
            for idx, c in enumerate(selected_clients_id):
                if results[idx]:  # Check if we got a valid result
                    # Load the trained weights back to the original model
                    models['clients'][c].load_state_dict(results[idx])
                    local_states.append(copy.deepcopy(results[idx]))
                else:
                    print(f"Warning: No valid result for client {c}, using original state")
                    local_states.append(copy.deepcopy(models['clients'][c].state_dict()))

            # Aggregation (identical to original code)
            selected_data_num = data_num[selected_clients_id]
            model_state = local_states[0]

            for key in local_states[0]:
                model_state[key] = model_state[key]*selected_data_num[0]
                for i in range(1, len(selected_clients_id)):
                    model_state[key] = model_state[key].float() + local_states[i][key].float() * selected_data_num[i]
                model_state[key] = model_state[key].float() / np.sum(selected_data_num)
            
            models['server'].load_state_dict(model_state, strict=False)
            
            print(f'Communication round: {com + 1}/{COMMUNICATION} | Cycle: {cycle + 1}/{CYCLES}')

    print('>> Training Finished.')

##
# Main
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  # Required for CUDA tensors sharing
    
    accuracies = [[] for _ in range(TRIALS)]

    indices = list(range(NUM_TRAIN))
    id2lab = []
    for id in indices:
        id2lab.append(cifar10_train[id][1])
    id2lab = np.array(id2lab)

    print(f"Generating Dirichlet partition with alpha {ALPHA}, seed {SEED} for {CLIENTS} clients...")

    data_splits = dirichlet_balanced_partition(cifar10_train, CLIENTS, alpha=ALPHA, seed=SEED)

    for trial in range(TRIALS):
        trial_seed = SEED + TRIAL_SEED_OFFSET * (trial + 1)
        set_all_seeds(trial_seed)

        print(f"\n=== Trial {trial+1}/{TRIALS} (Seed: {trial_seed}) ===\n")

        labeled_set_list = []
        unlabeled_set_list = []
        pseudo_set = []

        rest_data = indices.copy()

        print('Query Strategy:', ACTIVE_LEARNING_STRATEGY)
        print('Number of clients:', CLIENTS)
        print('Number of epochs:', EPOCH)
        print('Number of communication rounds:', COMMUNICATION)

        num_per_client = int(len(rest_data) / CLIENTS)
        resnet8 = resnet.preact_resnet8_cifar(num_classes=10)
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
            client_worker_init_fn = get_seed_worker(trial_seed + c * 100)

            # For labeled data loaders
            g_labeled = torch.Generator()
            g_labeled.manual_seed(trial_seed + c * 100 + 10000)

            # For unlabeled data loaders
            g_unlabeled = torch.Generator()
            g_unlabeled.manual_seed(trial_seed + c * 100 + 20000)

            data_list.append(data_splits[c])

            # Create a separate random generator that won't affect the rest of the randomness
            init_sample_rng = np.random.RandomState(trial_seed + c * 100)

            # Generate reproducible shuffled indices
            shuffled_indices = np.arange(len(data_splits[c]))
            init_sample_rng.shuffle(shuffled_indices)

            # Apply the shuffled indices
            data_list[c] = [data_splits[c][i] for i in shuffled_indices]

            labeled_set_list.append(data_list[c][:base[c]])

            values, counts = np.unique(id2lab[np.array(data_list[c])], return_counts=True)
            dictionary = dict(zip(values, counts))
            ratio = np.zeros(10)  # num_classes = 10
            ratio[values] = counts
            ratio /= np.sum(counts)
            data_ratio_list.append(ratio)

            values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
            ratio = np.zeros(10)  # num_classes = 10
            ratio[values] = counts
            loss_weight_list.append(ratio)

            data_num.append(len(labeled_set_list[c]))
            unlabeled_set_list.append(data_list[c][base[c]:])

            client_models.append(copy.deepcopy(resnet8).to(device))
        
        data_num = np.array(data_num)

        # added strategy manager
        strategy_manager = StrategyManager(
            strategy_name=ACTIVE_LEARNING_STRATEGY, 
            loss_weight_list=[torch.tensor(w, dtype=torch.float32).to(device) for w in loss_weight_list],
            device=device
        )

        del resnet8

        test_generator = torch.Generator()
        test_generator.manual_seed(trial_seed)  # Using the trial seed for consistency
        test_worker_init_fn = get_seed_worker(trial_seed + 50000)  # Unique seed for test loader
        test_loader = DataLoader(
            cifar10_test, 
            batch_size=BATCH,
            worker_init_fn=test_worker_init_fn, 
            generator=test_generator,
            pin_memory=True
        )

        dataloaders = {
            'labeled_sets': labeled_set_list,
            'unlabeled_sets': unlabeled_set_list,
            'test': test_loader
        }
        
        # Model
        models = {'clients': client_models, 'server': None}

        torch.backends.cudnn.benchmark = False

        print("Local model training using", LOCAL_MODEL_UPDATE)

        # Active learning cycles
        for cycle in range(CYCLES):
            server = resnet.preact_resnet8_cifar(num_classes=10).to(device)
            models['server'] = server
            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_clients = []
            sched_clients = []

            for c in range(CLIENTS):
                optim_clients.append(optim.SGD(models['clients'][c].parameters(), lr=LR,
                                   momentum=MOMENTUM, weight_decay=WDECAY))
                sched_clients.append(lr_scheduler.MultiStepLR(optim_clients[c], milestones=MILESTONES))

            optim_server = optim.SGD(models['server'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)

            sched_server = lr_scheduler.MultiStepLR(optim_server, milestones=MILESTONES)

            optimizers = {'clients': optim_clients, 'server': optim_server}
            schedulers = {'clients': sched_clients, 'server': sched_server}

            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, trial_seed)

            total_labels = 0
            for c in range(CLIENTS):
                total_labels += len(labeled_set_list[c])

            data_num = []
            loss_weight_list_2 = []
            server_state_dict = models['server'].state_dict()

            # Calculate sampling scores and sample for annotations.
            for c in range(CLIENTS): 
                cycle_worker_init_fn = get_seed_worker(trial_seed + c * 100 + cycle * 1000)
                g_labeled_cycle = torch.Generator()
                g_labeled_cycle.manual_seed(trial_seed + c * 100 + cycle * 1000 + 10000)
                g_unlabeled_cycle = torch.Generator()
                g_unlabeled_cycle.manual_seed(trial_seed + c * 100 + cycle * 1000 + 20000)
            
                c_rng = np.random.RandomState(trial_seed + c * 500 + cycle * 50)
                unlabeled_indices = np.array(unlabeled_set_list[c])
                c_rng.shuffle(unlabeled_indices)
                unlabeled_set_list[c] = unlabeled_indices.tolist()

                unlabeled_loader = DataLoader(cifar10_select, batch_size=BATCH,
                                             sampler=SubsetSequentialSampler(unlabeled_set_list[c]),
                                             num_workers=0, 
                                             worker_init_fn=cycle_worker_init_fn,
                                             generator=g_unlabeled_cycle,
                                             pin_memory=True)

                selected_samples, remaining_unlabeled = strategy_manager.select_samples(
                    models['clients'][c],
                    models['server'],
                    unlabeled_loader,
                    c,
                    unlabeled_set_list[c],
                    add[c],
                    seed=trial_seed + c * 100 + cycle * 1000
                )

                models['clients'][c].load_state_dict(server_state_dict, strict=False)

                # Update labeled and unlabeled sets
                labeled_set_list[c].extend(selected_samples)
                unlabeled_set_list[c] = remaining_unlabeled

                values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
                ratio = np.zeros(10)  # num_classes = 10
                ratio[values] = counts

                # compute new distributions
                loss_weight_list_2.append(ratio)
                data_num.append(len(labeled_set_list[c]))

            # evaluation
            acc_server = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Labelled sets size {}: server acc {}'.format(
                trial + 1, TRIALS, cycle + 1, CYCLES, total_labels, acc_server))

            # update distributions
            loss_weight_list = loss_weight_list_2
            dataloaders['labeled_sets'] = labeled_set_list
            dataloaders['unlabeled_sets'] = unlabeled_set_list
            data_num = np.array(data_num)
            accuracies[trial].append(acc_server)
            
        print('accuracies for trial {}:'.format(trial), accuracies[trial])

    print('accuracies means:', np.array(accuracies).mean(1))