import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from query_strategies.noise_stability import NoiseStabilitySampler
import copy

# Dummy Model (Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        out = self.fc2(features)
        return out, features  # Output and feature representation

# Dummy Dataset (MNIST-like)
def generate_dummy_data(num_samples=100):
    X = torch.randn(num_samples, 1, 28, 28)  # Fake grayscale images
    Y = torch.randint(0, 10, (num_samples,))  # Fake labels
    dataset = TensorDataset(X, Y)
    return dataset

# Initialize model and test dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
dataset = generate_dummy_data(100)
data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

# Initialize Noise Stability Sampler
sampler = NoiseStabilitySampler(device=device, noise_scale=0.001, num_sampling=5)

# ========== 1️⃣ TEST: Check Noise Perturbation ========== #
def test_noise_addition():
    print("🔹 TEST 1: Checking if noise is added correctly to model weights...")
    model_copy = copy.deepcopy(model)
    sampler.add_noise_to_weights(model_copy)

    for (orig_param, noisy_param) in zip(model.parameters(), model_copy.parameters()):
        if not torch.equal(orig_param, noisy_param):
            print("✅ Noise added successfully to model weights!")
            return
    print("❌ No difference detected after adding noise! Check implementation.")

test_noise_addition()

# ========== 2️⃣ TEST: Validate Feature Deviation ========== #
def test_feature_deviation():
    print("\n🔹 TEST 2: Checking if feature deviation is correctly computed...")
    
    # Get features before noise
    outputs = sampler.get_all_outputs(model, data_loader, use_feature=True)

    # Get features after adding noise
    noisy_model = copy.deepcopy(model)
    sampler.add_noise_to_weights(noisy_model)
    outputs_noisy = sampler.get_all_outputs(noisy_model, data_loader, use_feature=True)

    deviation = torch.norm(outputs_noisy - outputs, dim=1).mean().item()
    print(f"Feature deviation norm: {deviation:.6f}")

    if deviation > 0:
        print("✅ Feature deviation is correctly computed!")
    else:
        print("❌ Feature deviation is zero. Check perturbation implementation.")

test_feature_deviation()

# ========== 3️⃣ TEST: Sample Selection ========== #
def test_sample_selection():
    print("\n🔹 TEST 3: Checking if sample selection works correctly...")
    
    # Run sample selection
    unlabeled_set = list(range(100))  # Dummy sample indices
    selected_samples, remaining_unlabeled = sampler.select_samples(model, data_loader, unlabeled_set, num_samples=10)

    # Validate outputs
    print(f"🔹 Selected Samples: {selected_samples[:5]} ...")  # Print first 5 selected
    print(f"🔹 Remaining Unlabeled: {remaining_unlabeled[:5]} ...")  # Print first 5 remaining

    if len(selected_samples) == 10 and len(set(selected_samples).intersection(remaining_unlabeled)) == 0:
        print("✅ Sample selection works correctly!")
    else:
        print("❌ Sample selection has issues. Check sorting and indexing.")

test_sample_selection()
