import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# --- Step 1: Load partitioned data from JSON ---
filename = "alpha0-1_cifar10_20clients.json"
with open(filename, "r") as f:
    partitioned_data = json.load(f)

# --- Step 2: Download CIFAR-10 dataset to get the targets/labels ---
data_dir = "../data/"
train_dataset = datasets.CIFAR10(data_dir, train=True, download=True)

# --- Step 3: Compute label distribution for each client ---
num_clients = len(partitioned_data)
num_classes = 10

client_class_counts = np.zeros((num_clients, num_classes), dtype=int)

for client_idx, indices in enumerate(partitioned_data):
    for idx in indices:
        label = train_dataset.targets[idx]
        client_class_counts[client_idx, label] += 1

# --- Step 4: Plot the label distribution with smaller circles ---
fig, ax = plt.subplots(figsize=(12, 6))

# Reduced scaling factor for dot sizes
scaling_factor = 0.5

for client in range(num_clients):
    for c in range(num_classes):
        count = client_class_counts[client, c]
        if count > 0:
            ax.scatter(client, c, s=count * scaling_factor, alpha=0.6, edgecolors='w')

ax.set_xlabel('Client')
ax.set_ylabel('Class')
ax.set_title('Label Distribution Across Clients')
ax.set_xticks(range(num_clients))
ax.set_yticks(range(num_classes))
plt.grid(True)
plt.show()


# --- Step 3: Compute total number of labels (samples) per client ---
# Here, each client's count is simply the length of the list of indices
total_samples_per_client = [len(client_indices) for client_indices in partitioned_data]

# --- Step 4: Plot a histogram of total samples per client ---
plt.figure(figsize=(10, 6))
plt.hist(total_samples_per_client, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Labels (Samples) per Client')
plt.ylabel('Number of Clients')
plt.title('Distribution of Total Samples per Client')
plt.show()