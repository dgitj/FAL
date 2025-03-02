# Python
import os
import random
import copy
import json
import time
import importlib

# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.distributions import Beta

# mps_available = torch.backends.mps.is_available()
# print(f"MPS available: {mps_available}")

# cuda_available = torch.cuda.is_available()
# print(f"CUDA available: {cuda_available}")

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
        print('target_weight',target_weight.size())
        print('diff',diff.size())
        return torch.log(target_weight).unsqueeze(0).repeat(x.size(0),1) - diff - logsumexp

    def forward(self, logits, target, loss_weights):
        log_probabilities = self.log_softmax_weighted(logits, target = target, loss_weights = loss_weights)

        return -self.class_weights.index_select(0, target) * log_probabilities.index_select(-1, target).diag()

def read_data(dataloader):
    while True:
        for data in dataloader:
            yield data


iters = 0

"""
def train_epoch_client_distil(selected_clients_id, models, criterion, optimizers, dataloaders, comm):
    global iters

    kld = nn.KLDivLoss(reduce=False)

    for c in selected_clients_id:
        mod = models['clients'][c]
        mod.train()
        unlab_set = read_data(dataloaders['unlab-private'][c])
        for data in tqdm(dataloaders['train-private'][c], leave=False, total=len(dataloaders['train-private'][c])):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            unlab_data = next(unlab_set)
            unlab_inputs = unlab_data[0].to(device)

            # mix unlabelled data
            m = Beta(torch.FloatTensor([BETA[0]]).item(), torch.FloatTensor([BETA[1]]).item())
            beta_0 = m.sample(sample_shape=torch.Size([unlab_inputs.size(0)])).to(device)
            beta = beta_0.view(unlab_inputs.size(0), 1, 1, 1)
            indices = np.random.choice(unlab_inputs.size(0), replace=False)
            mixed_inputs =  beta * unlab_inputs + (1 - beta) * unlab_inputs[indices,...]

            scores, _  = mod(inputs)

            iters += 1
            optimizers['clients'][c].zero_grad()

            with torch.no_grad():
                scores_unlab_t, _ = models['server'](mixed_inputs)

            scores_unlab, _ = mod(mixed_inputs)
            _, pred_labels = torch.max((scores_unlab_t.data), 1)

            # find the \Gamma weight matrix
            mask = (loss_weight_list[c] > 0).float()
            weight_ratios = loss_weight_list[c].sum() / (loss_weight_list[c] + 1e-6)
            weight_ratios *= mask
            weights =  beta_0 * (weight_ratios / weight_ratios.sum())[pred_labels] + (1-beta_0)*(weight_ratios / weight_ratios.sum())[pred_labels[indices]]

            # compensatory loss
            distil_loss = int(comm>0)*(weights*kld(F.log_softmax(scores_unlab, -1), F.softmax(scores_unlab_t.detach(), -1)).mean(1)).mean()

            spc = loss_weight_list[c]
            spc = spc.unsqueeze(0).expand(labels.size(0), -1)
            scores = scores + spc.log()

            # client loss
            loss = torch.sum(criterion(scores, labels)) / labels.size(0)

            # KCFU loss
            (loss+ distil_loss).backward()
            optimizers['clients'][c].step()

            # log
            if (iters % 1000 == 0):
                print('loss: ', loss.item(), 'distil loss: ', distil_loss.item())
"""

def train_epoch_client_traditional(selected_clients_id, models, criterion, optimizers, dataloaders):
    global iters

    for c in selected_clients_id:
        mod = models['clients'][c]
        mod.train()
        
        for data in tqdm(dataloaders['train-private'][c], leave=False, total=len(dataloaders['train-private'][c])):
            inputs = data[0].cuda()
            labels = data[1].cuda()

            # Forward pass
            scores, _ = mod(inputs)
            
            # Standard cross-entropy loss without KCFU
            loss = torch.sum(criterion(scores, labels)) / labels.size(0)
            
            # Backward and optimize
            iters += 1
            optimizers['clients'][c].zero_grad()
            loss.backward()
            optimizers['clients'][c].step()
            
            # Log occasionally
            if (iters % 1000 == 0):
                print('loss: ', loss.item())
                

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


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs):
    print('>> Train a Model.')

    for com in range(COMMUNICATION):
        ###
        if com < COMMUNICATION-1:
            selected_clients_id = np.random.choice(CLIENTS, int(CLIENTS * RATIO), replace=False)
        else:
            selected_clients_id = range(CLIENTS)
        ###
        # ALTERNATIVE replace the above block with: selected_clients_id = np.random.choice(CLIENTS, int(CLIENTS * RATIO), replace=False)

        server_state_dict = models['server'].state_dict()

        # broadcast
        for c in selected_clients_id:
            models['clients'][c].load_state_dict(server_state_dict, strict=False)

        # local updates
        start = time.time()
        for epoch in range(num_epochs):
            train_epoch_client_traditional(selected_clients_id, models, criterion, optimizers, dataloaders)
            for c in selected_clients_id:
                schedulers['clients'][c].step() #changed order of optimizer and scheduler
            print(f'Epoch: {epoch + 1}/{num_epochs} | Communication round: {com + 1}/{COMMUNICATION} | Cycle: {cycle + 1}/{CYCLES}')    
        end = time.time()
        print('time epoch:',(end-start)/num_epochs)


        # aggregation
        local_states = [
            copy.deepcopy(models['clients'][c].state_dict())
            for c in selected_clients_id
        ]

        selected_data_num = data_num[selected_clients_id]
        model_state = local_states[0]

        for key in local_states[0]:
            model_state[key] = model_state[key]*selected_data_num[0]
            for i in range(1, len(selected_clients_id)):
                # model_state[key] += local_states[i][key]*selected_data_num[i]
                model_state[key] = model_state[key].float() + local_states[i][key].float() * selected_data_num[i]
            # model_state[key] /= np.sum(selected_data_num)
            model_state[key] = model_state[key].float() / np.sum(selected_data_num)
        models['server'].load_state_dict(model_state, strict=False)

    print('>> Finished.')


##
# Main
if __name__ == '__main__':

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
        resnet8 = resnet.preact_resnet8_cifar(num_classes=num_classes)
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
            # public_set.extend(data_list[c][-base:])
            labeled_set_list.append(data_list[c][:base[c]])
            values, counts = np.unique(id2lab[np.array(data_list[c])], return_counts=True)
            dictionary = dict(zip(values, counts))
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            ratio /= np.sum(counts)
            data_ratio_list.append(ratio)
            print('client {}, ratio {}'.format(c, ratio))

            values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
            ratio = np.zeros(num_classes)
            ratio[values] = counts
            loss_weight_list.append(torch.tensor(ratio, dtype=torch.float32).to(device))
            # loss_weight_list.append(torch.tensor(ratio).to(device).float())

            data_num.append(len(labeled_set_list[c]))
            unlabeled_set_list.append(data_list[c][base[c]:])
            private_train_loaders.append(DataLoader(cifar10_train, batch_size=BATCH,
                           sampler=SubsetRandomSampler(labeled_set_list[c]),
                           # num_workers= 4,
                            num_workers= 2,
                           pin_memory=True))
            private_unlab_loaders.append(DataLoader(cifar10_train, batch_size=BATCH,
                                                    sampler=SubsetRandomSampler(unlabeled_set_list[c]),
                                                    num_workers=4,
                                                    pin_memory=True))

            client_models.append(copy.deepcopy(resnet8).to(device))
        data_num = np.array(data_num)


        # added strategy manager
        strategy_manager = StrategyManager(
            strategy_name=ACTIVE_LEARNING_STRATEGY, 
            loss_weight_list=loss_weight_list,
            device=device
        )


        del resnet8

        test_loader  = DataLoader(cifar10_test, batch_size=BATCH)

        dataloaders  = {'train-private': private_train_loaders,'unlab-private': private_unlab_loaders, 'test': test_loader}
        
        # Model

        models      = {'clients': client_models, 'server': None}

        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):

            server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
            models['server'] = server
            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_clients = []
            sched_clients = []

            for c in range(CLIENTS):
                optim_clients.append(optim.SGD(models['clients'][c].parameters(), lr=LR,
                                    momentum=MOMENTUM, weight_decay=WDECAY))
                sched_clients.append(lr_scheduler.MultiStepLR(optim_clients[c], milestones=MILESTONES))

            optim_server  = optim.SGD(models['server'].parameters(), lr=LR,
                                        momentum=MOMENTUM, weight_decay=WDECAY)


            sched_server = lr_scheduler.MultiStepLR(optim_server, milestones=MILESTONES)

            optimizers = {'clients': optim_clients, 'server': optim_server}
            schedulers = {'clients': sched_clients, 'server': sched_server}

            unlabeled_loaders = []

            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH)

            total_labels = 0
            for c in range(CLIENTS):
                total_labels += len(labeled_set_list[c])

            private_train_loaders = []
            data_num = []
            loss_weight_list_2 = []
            server_state_dict = models['server'].state_dict()

            # Calculate sampling scores and sample for annotations.
            for c in range(CLIENTS):
                random.shuffle(unlabeled_set_list[c])
                unlabeled_loader = DataLoader(cifar10_select, batch_size=BATCH,
                                              sampler=SubsetSequentialSampler(unlabeled_set_list[c]),
                                              pin_memory=True)

                selected_samples, remaining_unlabeled = strategy_manager.select_samples(
                    models['clients'][c],               # Client model
                    models['server'],                   # Server model (only used for discrepancy)
                    unlabeled_loader,                   # Unlabeled data for the client
                    c,                                  # Client ID (only for discrepancy)
                    unlabeled_set_list[c],              # List of unlabeled sample IDs
                    add[c]                              # Number of samples to select
                )

                models['clients'][c].load_state_dict(server_state_dict, strict=False)

                # Update labeled and unlabeled sets
                labeled_set_list[c].extend(selected_samples)
                unlabeled_set_list[c] = remaining_unlabeled

                values, counts = np.unique(id2lab[np.array(labeled_set_list[c])], return_counts=True)
                ratio = np.zeros(num_classes)

                ratio[values] = counts

                # compute new distributions
                loss_weight_list_2.append(torch.tensor(ratio).to(device).float())
                data_num.append(len(labeled_set_list[c]))

                # dataloaders with updated data pools
                private_train_loaders.append(DataLoader(cifar10_train, batch_size=BATCH,
                                                        sampler=SubsetRandomSampler(labeled_set_list[c]),
                                                        num_workers=4,
                                                        pin_memory=True))
                private_unlab_loaders.append(DataLoader(cifar10_train, batch_size=BATCH,
                                                        sampler=SubsetRandomSampler(unlabeled_set_list[c]),
                                                        num_workers=4,
                                                        pin_memory=True))
            # evaluation
            acc_server = test(models, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Labelled sets size {}: server acc {}'.format(
                trial + 1, TRIALS, cycle + 1, CYCLES, total_labels, acc_server))

            # update distributions
            loss_weight_list = loss_weight_list_2
            dataloaders['train-private'] = private_train_loaders
            dataloaders['unlab-private'] = private_unlab_loaders
            data_num = np.array(data_num)
            accuracies[trial].append(acc_server)
        print('accuracies for trial {}:'.format(trial), accuracies[trial])

    print('accuracies means:',np.array(accuracies).mean(1))