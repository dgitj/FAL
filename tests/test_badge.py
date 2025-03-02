import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from query_strategies.badge import BADGESampler
import copy

# Dummy Model (Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        out = self.fc2(features)
        return out  # No softmax, as we use logits for gradient embeddings

# Dummy Dataset (MNIST-like)
def generate_dummy_data(num_samples=100, num_classes=10):
    X = torch.randn(num_samples, 1, 28, 28)  # Fake grayscale images
    Y = torch.randint(0, num_classes, (num_samples,))  # Fake labels
    dataset = TensorDataset(X, Y)
    return dataset

# Initialize model and test dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
dataset = generate_dummy_data(100)
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Use larger batch size

# Initialize BADGE Sampler
sampler = BADGESampler(device=device)

# ========== 1️⃣ TEST: Check Gradient Embeddings Computation ========== #
def test_gradient_computation():
    print("🔹 TEST 1: Checking gradient embedding computation...")
    
    # Compute gradient embeddings
    gradients, data_indices = sampler.compute_gradient_embeddings(model, data_loader)

    # Verify gradients
    print(f"Gradient Shape: {gradients.shape}, Data Indices Length: {len(data_indices)}")
    
    if gradients.nelement() == 0:
        print("❌ ERROR: No gradients computed!")
    elif np.isnan(gradients.numpy()).any():
        print("❌ ERROR: NaN detected in gradients!")
    else:
        print("✅ Gradients computed successfully!")

test_gradient_computation()

# ========== 2️⃣ TEST: Sample Selection ========== #
def test_sample_selection():
    print("\n🔹 TEST 2: Checking if sample selection works correctly...")
    
    # Create dummy indices for unlabeled set
    unlabeled_set = list(range(100))
    
    # Select samples based on gradient embeddings
    selected_samples, remaining_unlabeled = sampler.select_samples(model, data_loader, unlabeled_set, num_samples=10)

    # Validate outputs
    print(f"🔹 Selected Samples: {selected_samples[:5]} ...")  # Print first 5 selected
    print(f"🔹 Remaining Unlabeled: {remaining_unlabeled[:5]} ...")  # Print first 5 remaining

    if len(selected_samples) == 10 and len(set(selected_samples).intersection(remaining_unlabeled)) == 0:
        print("✅ Sample selection works correctly!")
    else:
        print("❌ Sample selection has issues. Check sorting and indexing.")

test_sample_selection()
