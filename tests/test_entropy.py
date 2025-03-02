import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from query_strategies.entropy import EntropySampler
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
        return out  # No softmax, as entropy is calculated later

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

# Initialize Entropy Sampler
sampler = EntropySampler(device=device)


def train_dummy_model(model, dataset, device, epochs=2):
    """Trains the dummy model to make entropy values more meaningful."""
    model.train()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()  # Set back to eval mode
    print(f"✅ Model trained for {epochs} epochs.")

# Before running tests, train the model
train_dummy_model(model, dataset, device)


# ========== 1️⃣ TEST: Check Entropy Computation ========== #
def test_entropy_computation():
    print("🔹 TEST 1: Checking entropy computation...")
    
    # Create dummy indices for unlabeled set
    unlabeled_set = list(range(100))
    
    # Compute entropy scores
    entropy_scores = sampler.compute_entropy(model, data_loader, unlabeled_set)

    # Verify entropy values
    print(f"Entropy Scores Range: min={np.min(entropy_scores):.4f}, max={np.max(entropy_scores):.4f}, mean={np.mean(entropy_scores):.4f}")

    if np.any(entropy_scores < 0):
        print("❌ ERROR: Negative entropy detected!")
    elif np.all(entropy_scores == entropy_scores[0]):
        print("❌ ERROR: All entropy scores are identical, check variance!")
    else:
        print("✅ Entropy scores computed successfully!")

test_entropy_computation()

# ========== 2️⃣ TEST: Sample Selection ========== #
def test_sample_selection():
    print("\n🔹 TEST 2: Checking if sample selection works correctly...")
    
    # Create dummy indices for unlabeled set
    unlabeled_set = list(range(100))
    
    # Select samples based on entropy
    selected_samples, remaining_unlabeled = sampler.select_samples(model, data_loader, unlabeled_set, num_samples=10)

    # Validate outputs
    print(f"🔹 Selected Samples: {selected_samples[:5]} ...")  # Print first 5 selected
    print(f"🔹 Remaining Unlabeled: {remaining_unlabeled[:5]} ...")  # Print first 5 remaining

    if len(selected_samples) == 10 and len(set(selected_samples).intersection(remaining_unlabeled)) == 0:
        print("✅ Sample selection works correctly!")
    else:
        print("❌ Sample selection has issues. Check sorting and indexing.")

test_sample_selection()

# ========== 3️⃣ TEST: Edge Cases ========== #
def test_edge_cases():
    print("\n🔹 TEST 3: Checking edge cases...")

    # Edge case 1: Empty dataset
    empty_loader = DataLoader([], batch_size=10)
    try:
        empty_entropy = sampler.compute_entropy(model, empty_loader, [])
        print("❌ ERROR: Computed entropy for empty dataset!")
    except Exception as e:
        print("✅ Correctly handled empty dataset:", str(e))

    # Edge case 2: Very low entropy variance
    def fake_low_entropy_model(x):
        return torch.full((x.size(0), 10), -2.0).to(device)  # Low variance outputs

    model.forward = fake_low_entropy_model  # Override model for test
    low_entropy_scores = sampler.compute_entropy(model, data_loader, unlabeled_set)

    if np.var(low_entropy_scores) < 1e-5:
        print("✅ Correctly identified low entropy variance issue!")
    else:
        print("❌ ERROR: Low variance case not handled!")

test_edge_cases()
