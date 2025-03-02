import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from query_strategies.feal import FEALSampler
import copy

# Dummy Model (Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        out = self.fc2(features)
        block_outputs = [features]
        return out, torch.softmax(out, dim=-1), features, block_outputs

# Dummy Dataset
def generate_dummy_data(num_samples=100):
    X = torch.randn(num_samples, 1, 28, 28)  # Fake grayscale images
    Y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, Y)
    return dataset

# Mock Args
class Args:
    batch_size = 32
    n_neighbor = 5
    cosine = 0.85

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
global_model = SimpleCNN().to(device)
dataset = generate_dummy_data(100)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
args = Args()
sampler = FEALSampler(device=device)

# ========== 1️⃣ TEST: Discrepancy Computation ========== #
def test_discrepancy_computation():
    print("🔹 TEST 1: Checking discrepancy computation...")

    unlabeled_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    g_data, l_data, g_dis, l_features = sampler.compute_discrepancy(global_model, model, unlabeled_loader)

    print(f"Global uncertainty min/max/mean: {g_data.min():.4f}/{g_data.max():.4f}/{g_data.mean():.4f}")
    print(f"Local uncertainty min/max/mean: {l_data.min():.4f}/{l_data.max():.4f}/{l_data.mean():.4f}")

    if torch.all(g_data == 0) or torch.all(l_data == 0):
        print("❌ ERROR: Zero uncertainties detected!")
    elif torch.isnan(g_data).any() or torch.isnan(l_data).any():
        print("❌ ERROR: NaNs detected in uncertainties!")
    else:
        print("✅ Discrepancy computed successfully!")

test_discrepancy_computation()

# ========== 2️⃣ TEST: Sample Selection ========== #
def test_sample_selection():
    print("\n🔹 TEST 2: Checking sample selection...")

    unlabeled_set = list(range(100))
    
    selected_samples, remaining_unlabeled = sampler.select_samples(
        global_model=global_model,
        local_model=model,
        unlabeled_loader=data_loader,  # Pass the actual dataloader
        unlabeled_set=unlabeled_set,
        num_samples=10,  # ✅ Correct parameter name
        args=args
    )

    print(f"Selected Samples: {selected_samples[:5]} ...")
    print(f"Remaining Unlabeled: {remaining_unlabeled[:5]} ...")

    if len(selected_samples) == 10 and len(set(selected_samples).intersection(remaining_unlabeled)) == 0:
        print("✅ Sample selection works correctly!")
    else:
        print("❌ ERROR: Issue in sample selection.")

test_sample_selection()

# ========== 4️⃣ TEST: Model Usage ========== #
def test_model_usage():
    print("\n🔹 TEST 3: Ensuring both local and global models are used...")

    def frozen_global_model(x):
        batch_size = x.size(0)
        return torch.zeros(batch_size, 10).to(device), None, None, [torch.zeros(batch_size, 128).to(device)]

    global_model.forward = frozen_global_model

    unlabeled_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    g_data, l_data, g_dis, l_features = sampler.compute_discrepancy(global_model, model, unlabeled_loader)

    if torch.allclose(g_data, g_data[0], atol=1e-4):
        print("✅ Frozen global model produces consistent uncertainty (as expected for uniform predictions).")
    else:
        print("❌ ERROR: Frozen global model produced inconsistent uncertainties.")

test_model_usage()

# ========== 4️⃣ TEST: Edge Cases ========== #
def test_edge_cases():
    print("\n🔹 TEST 4: Checking edge cases...")

    # Edge case: identical models
    identical_discrepancy = sampler.compute_discrepancy(model, model, DataLoader(dataset, batch_size=32))

    if torch.all(identical_discrepancy[0] == identical_discrepancy[1]):
        print("✅ Identical models produce matching uncertainties.")
    else:
        print("❌ ERROR: Identical models produced differing uncertainties.")

test_edge_cases()
