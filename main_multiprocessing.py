import os
import random
import copy
import json
import time
import importlib
import functools

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

# Torchvision
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

# Utils
from config import *
from tqdm import tqdm

import multiprocessing as mp

def debug_dataloader(loader, client_id, tag="train"):
    """Print checksum and first few samples to verify determinism."""
    samples = []
    for i, (inputs, labels) in enumerate(loader):
        if i >= 3:  # Just check first 3 batches
            break
        # Convert to numpy for consistent output
        batch_samples = [(idx, lab.item()) for idx, lab in 
                         zip(range(len(labels)), labels)]
        samples.append(batch_samples)
        print(f"Client {client_id}, {tag} batch {i}: First 5 samples: {batch_samples[:5]}")
    
    # Calculate a checksum of all indices and labels to verify consistency
    flat_samples = [item for batch in samples for item in batch]
    checksum = sum([idx * 10 + lab for idx, lab in flat_samples]) % 10000
    print(f"Client {client_id}, {tag} data checksum: {checksum}")
    return checksum

def model_checksum(model):
    """Calculate a simple checksum of model weights for comparison."""
    state_dict = model.state_dict()
    checksum = 0
    for key in sorted(state_dict.keys()):
        # Get a simple numeric representation of each parameter tensor
        param_sum = state_dict[key].abs().sum().item()
        # Add to checksum with a multiplier based on parameter name
        # This ensures different parameters contribute differently
        checksum += param_sum * hash(key) % 10000
    return checksum % 1000000


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

def get_seed_worker(base_seed):
    """Creates a worker initialization function with the given base seed"""
    return functools.partial(seed_worker_fn, base_seed)

# Dataset-specific sampler
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

# Utility to cycle through dataloader indefinitely
def read_data(dataloader):
    while True:
        for data in dataloader:
            yield data

# Global iteration counter (for logging)
global_iters = 0

#########################################
# Per-client training function (parallel)
#########################################
def train_client(args):
    """
    Trains a single client for a number of epochs and returns a tuple (client_id, updated_state_dict).

    Expected tuple (16 elements):
      1. client_id: integer client id.
      2. update_type: string; "Vanilla" or "KCFU".
      3. comm_round: current communication round.
      4. base_trial_seed: trial_seed for this round.
      5. num_epochs: number of local epochs.
      6. initial_state_dict: state dict (from server) to load.
      7. loss_weight: torch tensor for loss weighting.
      8. lr: learning rate.
      9. momentum: momentum.
      10. weight_decay: weight decay.
      11. milestones: scheduler milestones.
      12. device: torch device.
      13. num_classes: number of classes.
      14. client_worker_seed: seed to construct DataLoaders.
      15. labeled_indices: list of indices for the client’s labeled data.
      16. unlabeled_indices: list of indices for the client’s unlabeled data.
    """
    (client_id, update_type, comm_round, base_trial_seed, num_epochs,
     initial_state_dict, loss_weight, lr, momentum, weight_decay, milestones,
     device, num_classes, client_worker_seed, labeled_indices, unlabeled_indices) = args

    # set_all_seeds(client_worker_seed)
    # Ensure deterministic behavior
      # SAVE original random state
    original_torch_state = torch.get_rng_state()
    if torch.cuda.is_available():
        original_cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    
    # SET EXACT SEED for model creation
    torch.manual_seed(base_trial_seed)  # NOT base_trial_seed or client_worker_seed!
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_trial_seed)

    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    # Create client model and load initial (server) state.
    client_model = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
    # Use strict=True and copy with float() conversion
    client_model.load_state_dict({k: v.clone() for k, v in initial_state_dict.items()}, strict=True)
    #client_model.load_state_dict(initial_state_dict, strict=False)
      # Restore the random states for other operations
      # RESTORE original random state
    torch.set_rng_state(original_torch_state)
    if torch.cuda.is_available():
        for i, state in enumerate(original_cuda_states):
            torch.cuda.set_rng_state(state, i)

    print(f"Parallel - Client {client_id} model checksum before training: {model_checksum(client_model)}")
    client_model.train()

    torch.manual_seed(base_trial_seed + client_id * 100 + 10000)
    np.random.seed(base_trial_seed + client_id * 100 + 10000)
    random.seed(base_trial_seed + client_id * 100 + 10000)
    
    # Create DataLoaders
    client_worker_init_fn = get_seed_worker(client_worker_seed)
    g_labeled = torch.Generator()
    g_labeled.manual_seed(base_trial_seed + client_id * 100 + 10000)

     # Create train loader with SubsetRandomSampler (exactly as in sequential)
    train_loader = DataLoader(
        cifar10_train, batch_size=BATCH,
        sampler=SubsetRandomSampler(labeled_indices),  # Same as sequential
        num_workers=0,
        worker_init_fn=client_worker_init_fn,
        generator=g_labeled,
        pin_memory=True
    )
    

    # Change to:
    if update_type != "Vanilla":
        # Pre-shuffle with deterministic RNG
        unlab_rng = random.Random(base_trial_seed + client_id * 100 + 20000)
        unlab_idx = unlabeled_indices.copy()
        unlab_rng.shuffle(unlab_idx)
        
        unlab_loader = DataLoader(
            cifar10_train, batch_size=BATCH,
            sampler=SubsetRandomSampler(unlabeled_indices),
            num_workers=0,
            worker_init_fn=client_worker_init_fn,
            generator=g_unlab,
            pin_memory=True
        )
    else:
        unlab_loader = None

    print(f"\n==== PARALLEL IMPLEMENTATION - CLIENT {client_id} ====")
    print(f"=== Debugging DataLoader for Client {client_id} ===")
    train_checksum = debug_dataloader(train_loader, client_id, "train")
    if update_type != "Vanilla" and unlab_loader:
        unlab_checksum = debug_dataloader(unlab_loader, client_id, "unlab")
    print(f"=======================================\n")

    # Optimizer and scheduler
    optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # For KCFU, prepare teacher model and unlabeled generator.
    if update_type != "Vanilla":
        batch_rng = np.random.RandomState(batch_seed)
        indices = batch_rng.choice(unlab_inputs.size(0), size=unlab_inputs.size(0), replace=False)
        kld = nn.KLDivLoss(reduction='none')
        teacher_model = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
        teacher_model.load_state_dict(initial_state_dict, strict=False)
        teacher_model.eval()
        unlab_gen = read_data(unlab_loader)

    local_iters = 0

    for epoch in range(num_epochs):
        # Use the same base_trial_seed for each epoch (as in sequential version)
        epoch_seed = int(base_trial_seed + epoch)
        if update_type == "Vanilla":
            client_seed = int(epoch_seed + client_id * 1000)
        else:
            client_seed = int(epoch_seed + client_id * 1000 + comm_round * 10)

        torch.manual_seed(client_seed)
        np.random.seed(client_seed)
        random.seed(client_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(client_seed)

        for batch_idx, data in enumerate(tqdm(train_loader, leave=False, total=len(train_loader))):
            batch_seed = int(client_seed + batch_idx)
            random.seed(batch_seed)
            np.random.seed(batch_seed)
            torch.manual_seed(batch_seed)
             
            if torch.cuda.is_available():
                torch.cuda.manual_seed(batch_seed)

            inputs = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()

            if update_type == "Vanilla":
                scores, _ = client_model(inputs)
                loss = torch.sum(criterion(scores, labels)) / labels.size(0)
                loss.backward()
            else:
                unlab_data = next(unlab_gen)
                unlab_inputs = unlab_data[0].to(device)
                m = Beta(torch.tensor([BETA[0]], dtype=torch.float32), torch.tensor([BETA[1]], dtype=torch.float32))
                beta_0 = m.sample(sample_shape=torch.Size([unlab_inputs.size(0)])).to(device)
                beta = beta_0.view(unlab_inputs.size(0), 1, 1, 1)
                batch_rng = np.random.RandomState(batch_seed)
                indices = batch_rng.choice(unlab_inputs.size(0), size=unlab_inputs.size(0), replace=False)
                mixed_inputs = beta * unlab_inputs + (1 - beta) * unlab_inputs[indices, ...]
                scores, _ = client_model(inputs)
                with torch.no_grad():
                    scores_unlab_t, _ = teacher_model(mixed_inputs)
                scores_unlab, _ = client_model(mixed_inputs)
                _, pred_labels = torch.max(scores_unlab_t.data, 1)
                mask = (loss_weight > 0).float()
                weight_ratios = loss_weight.sum() / (loss_weight + 1e-6)
                weight_ratios *= mask
                denom = weight_ratios.sum()
                weights = beta_0 * (weight_ratios / denom)[pred_labels] \
                          + (1 - beta_0) * (weight_ratios / denom)[pred_labels[indices]]
                distil_loss = int(comm_round > 0) * (weights *
                              kld(F.log_softmax(scores_unlab, -1),
                                  F.softmax(scores_unlab_t.detach(), -1)).mean(1)).mean()
                spc = loss_weight.unsqueeze(0).expand(labels.size(0), -1)
                scores = scores + spc.log()
                loss = torch.sum(criterion(scores, labels)) / labels.size(0)
                (loss + distil_loss).backward()

            optimizer.step()
            local_iters += 1
            if (local_iters % 1000 == 0):
                print(f"Client {client_id} | Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item()}")

        scheduler.step()
        print(f"Parallel - Client {client_id} model checksum after training: {model_checksum(client_model)}")

    return (client_id, client_model.state_dict())

#########################################
# Modified train function (multiprocessing)
#########################################
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, trial_seed):
    print('>> Train a Model.')
    for com in range(COMMUNICATION):
        rng = np.random.RandomState(trial_seed + com * 100)
        if com < COMMUNICATION - 1:
            selected_clients_id = rng.choice(CLIENTS, int(CLIENTS * RATIO), replace=False)
        else:
            selected_clients_id = list(range(CLIENTS))
        server_state_dict = models['server'].state_dict()
        for c in selected_clients_id:
            models['clients'][c].load_state_dict(server_state_dict, strict=False)

        args_list = []
        for c in selected_clients_id:
            # Construct the DataLoader inside the client process using the client's indices.
            # We pass the current labeled and unlabeled indices.
            args_list.append((
                int(c),
                LOCAL_MODEL_UPDATE,
                com,
                int(trial_seed),
                num_epochs,
                server_state_dict,            # initial state
                loss_weight_list[c],          # loss weight for client c
                LR,                           # learning rate
                MOMENTUM,
                WDECAY,
                MILESTONES,
                device,
                10,                           # num_classes (for CIFAR10)
                int(trial_seed + c * 100),         # client_worker_seed
                labeled_set_list[c],          # labeled indices for client c
                unlabeled_set_list[c]         # unlabeled indices for client c
            ))
        with mp.Pool(processes=len(selected_clients_id)) as pool:
            results = pool.map(train_client, args_list)
        
        print(f"Parallel - Server model checksum before aggregation: {model_checksum(models['server'])}")

        local_states = [state_dict for (_, state_dict) in results]
        selected_data_num = data_num[selected_clients_id]
        model_state = copy.deepcopy(local_states[0])
        for key in model_state:
            model_state[key] = model_state[key] * selected_data_num[0]
            for i in range(1, len(local_states)):
                model_state[key] = model_state[key].float() + local_states[i][key].float() * selected_data_num[i]
            model_state[key] = model_state[key].float() / np.sum(selected_data_num)
        models['server'].load_state_dict(model_state, strict=False)
        print(f"Parallel - Server model checksum after aggregation: {model_checksum(models['server'])}")
        print(f'Communication round: {com + 1}/{COMMUNICATION} completed.')
    print('>> Finished.')

#########################################
# Testing and main loop
#########################################
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

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    accuracies = [[] for _ in range(TRIALS)]
    indices = list(range(NUM_TRAIN))
    id2lab = [cifar10_train[i][1] for i in indices]
    id2lab = np.array(id2lab)
    print(f"Generating Dirichlet partition with alpha {ALPHA}, seed {SEED} for {CLIENTS} clients...")
    data_splits = dirichlet_balanced_partition(cifar10_train, CLIENTS, alpha=ALPHA, seed=SEED)

    for trial in range(TRIALS):
        trial_seed = int(SEED + TRIAL_SEED_OFFSET * (trial + 1))
        set_all_seeds(trial_seed)
        print(f"\n=== Trial {trial+1}/{TRIALS} (Seed: {trial_seed}) ===\n")
        labeled_set_list = []
        unlabeled_set_list = []
        pseudo_set = []
        private_train_loaders = []
        private_unlab_loaders = []
        num_classes = 10  # CIFAR10
        rest_data = indices.copy()
        print('Query Strategy:', ACTIVE_LEARNING_STRATEGY)
        print('Number of clients:', CLIENTS)
        print('Number of epochs:', EPOCH)
        print('Number of communication rounds:', COMMUNICATION)
        num_per_client = int(len(rest_data) / CLIENTS)
        resnet8 = resnet.preact_resnet8_cifar(num_classes=num_classes)
        client_models = []
        data_list = []
        total_data_num = []
        for c in range(CLIENTS):
            total_data_num.append(len(data_splits[c]))
            partition_checksum = sum(data_splits[c]) % 10000
            print(f"Client {c} initial partition checksum: {partition_checksum}")
        total_data_num = np.array(total_data_num)
        base = np.ceil((BASE / NUM_TRAIN) * total_data_num).astype(int)
        add = np.ceil((BUDGET / NUM_TRAIN) * total_data_num).astype(int)
        print('base number: ', base)
        print('budget each round: ', add)
        data_num = []
        data_ratio_list = []
        loss_weight_list = []
        for c in range(CLIENTS):
            client_worker_init_fn = get_seed_worker(trial_seed + c * 100)
            g_labeled = torch.Generator()
            g_labeled.manual_seed(trial_seed + c * 100 + 10000)
            g_unlabeled = torch.Generator()
            g_unlabeled.manual_seed(trial_seed + c * 100 + 20000)
            data_list.append(data_splits[c])
            init_sample_rng = np.random.RandomState(trial_seed + c * 100)
            shuffled_indices = np.arange(len(data_splits[c]))
            init_sample_rng.shuffle(shuffled_indices)
            data_list[c] = [data_splits[c][i] for i in shuffled_indices]
            shuffled_checksum = sum(data_list[c]) % 10000
            print(f"Client {c} shuffled partition checksum: {shuffled_checksum}")
            labeled_set_list.append(data_list[c][:base[c]])
            values, counts = np.unique(id2lab[np.array(data_list[c])], return_counts=True)
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            ratio /= np.sum(counts)
            data_ratio_list.append(ratio)
            values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            loss_weight_list.append(torch.tensor(ratio, dtype=torch.float32).to(device))
            data_num.append(len(labeled_set_list[c]))
            unlabeled_set_list.append(data_list[c][base[c]:])
            client_models.append(copy.deepcopy(resnet8).to(device))
        data_num = np.array(data_num)
        strategy_manager = StrategyManager(
            strategy_name=ACTIVE_LEARNING_STRATEGY,
            loss_weight_list=loss_weight_list,
            device=device
        )
        del resnet8
        test_generator = torch.Generator()
        test_generator.manual_seed(trial_seed)
        test_worker_init_fn = get_seed_worker(trial_seed + 50000)
        test_loader = DataLoader(
            cifar10_test,
            batch_size=BATCH,
            worker_init_fn=test_worker_init_fn,
            generator=test_generator,
            pin_memory=True
        )
        dataloaders = {'train-private': private_train_loaders, 'unlab-private': private_unlab_loaders, 'test': test_loader}
        models = {'clients': client_models, 'server': None}
        torch.backends.cudnn.benchmark = False
        print("Local model training using", LOCAL_MODEL_UPDATE)

        # Active learning cycles
        for cycle in range(CYCLES):
            torch.manual_seed(trial_seed)  # Same seed as sequential
            np.random.seed(trial_seed)
            random.seed(trial_seed)
            server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
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
            print(f"Cycle {cycle+1}/{CYCLES}: Starting local training ...")
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, trial_seed)
            total_labels = sum(len(labeled_set_list[c]) for c in range(CLIENTS))
            private_train_loaders = []
            private_unlab_loaders = []
            data_num_list = []
            loss_weight_list_2 = []
            server_state_dict = models['server'].state_dict()
            for c in range(CLIENTS):
                cycle_worker_init_fn = get_seed_worker(trial_seed + c * 100 + cycle * 1000)
                g_labeled_cycle = torch.Generator()
                g_labeled_cycle.manual_seed(trial_seed + c * 100 + cycle * 1000 + 10000)
                g_unlabeled_cycle = torch.Generator()
                g_unlabeled_cycle.manual_seed(trial_seed + c * 100 + cycle * 1000 + 20000)
                c_rng = np.random.RandomState(trial_seed + c * 500 + cycle * 50)
                unlabeled_indices_arr = np.array(unlabeled_set_list[c])
                c_rng.shuffle(unlabeled_indices_arr)
                unlabeled_set_list[c] = unlabeled_indices_arr.tolist()
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
                labeled_set_list[c].extend(selected_samples)
                unlabeled_set_list[c] = remaining_unlabeled
                values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
                ratio = np.zeros(num_classes)
                ratio[values] = counts
                loss_weight_list_2.append(torch.tensor(ratio).to(device).float())
                data_num_list.append(len(labeled_set_list[c]))
                private_train_loaders.append(DataLoader(cifar10_train, batch_size=BATCH,
                                                        sampler=SubsetRandomSampler(labeled_set_list[c]),
                                                        num_workers=0,
                                                        worker_init_fn=cycle_worker_init_fn,
                                                        generator=g_labeled_cycle,
                                                        pin_memory=True))
                private_unlab_loaders.append(DataLoader(cifar10_train, batch_size=BATCH,
                                                        sampler=SubsetRandomSampler(unlabeled_set_list[c]),
                                                        num_workers=0,
                                                        worker_init_fn=cycle_worker_init_fn,
                                                        generator=g_unlabeled_cycle,
                                                        pin_memory=True))
            acc_server = test(models, dataloaders, mode='test')
            print(f"Trial {trial+1}/{TRIALS} || Cycle {cycle+1}/{CYCLES} || Labeled set total: {total_labels} | Server acc: {acc_server}")
            loss_weight_list = loss_weight_list_2
            dataloaders['train-private'] = private_train_loaders
            dataloaders['unlab-private'] = private_unlab_loaders
            data_num = np.array(data_num_list)
            accuracies[trial].append(acc_server)
        print('Accuracies for trial {}:'.format(trial+1), accuracies[trial])
    print('Accuracies means:', np.array(accuracies).mean(1))
