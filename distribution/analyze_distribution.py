import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from scipy.stats import beta

# --- Load partitioned data from JSON ---
filename = "alpha0-1_cifar10_10clients.json"
with open(filename, "r") as f:
    partitioned_data = json.load(f)

# --- Download CIFAR-10 to obtain labels ---
data_dir = "../data/"
train_dataset = datasets.CIFAR10(data_dir, train=True, download=True)

num_clients = len(partitioned_data)
num_classes = 10

# --- Compute counts per client per class ---
client_class_counts = np.zeros((num_clients, num_classes), dtype=int)
for client_idx, indices in enumerate(partitioned_data):
    for idx in indices:
        label = train_dataset.targets[idx]
        client_class_counts[client_idx, label] += 1

# --- For each class, compute empirical proportions ---
empirical_proportions = {}  # dictionary: key=class, value=list of proportions for each client
for c in range(num_classes):
    # Total samples in class c (summing over all clients)
    total_c = client_class_counts[:, c].sum()
    # For each client, compute the proportion of class c
    if total_c > 0:
        empirical_proportions[c] = client_class_counts[:, c] / total_c
    else:
        empirical_proportions[c] = np.zeros(num_clients)

# --- Plot histograms and overlay theoretical Beta PDF for each class ---
fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()

# Theoretical Beta parameters for a Dirichlet with all parameters = 0.1
a_param = 0.1
b_param = (num_clients - 1) * 0.1

for c in range(num_classes):
    ax = axs[c]
    obs_props = empirical_proportions[c]
    
    # Plot histogram of the observed proportions for class c
    ax.hist(obs_props, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black', label="Observed")
    
    # Generate x values for plotting the Beta PDF
    x = np.linspace(0, max(obs_props) * 1.1, 200)
    y = beta.pdf(x, a_param, b_param)
    ax.plot(x, y, 'r-', lw=2, label=f"Beta PDF\nBeta({a_param}, {b_param:.1f})")
    
    ax.set_title(f"Class {c}")
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Density")
    ax.legend()

plt.tight_layout()
plt.show()
