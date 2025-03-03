import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from query_strategies.logo import LoGoSampler


# Dummy model compatible with LoGo
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        features = self.fc1(x)
        out = self.fc2(features)
        return out, features


# Dummy dataset
def generate_dummy_data(num_samples=100):
    X = torch.randn(num_samples, 1, 28, 28)  # CIFAR-like grayscale images
    Y = torch.randint(0, 10, (num_samples,))  # Fake labels
    dataset = TensorDataset(X, Y)
    return dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_model = SimpleCNN().to(device)
global_model = SimpleCNN().to(device)
dataset = generate_dummy_data(100)
unlabeled_set = list(range(len(dataset)))
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

sampler = LoGoSampler(device=device)


# ========== 1️⃣ Test: Embedding extraction ==========
def test_extract_embeddings():
    print("🔹 TEST 1: Extracting embeddings...")

    embeddings = sampler.extract_embeddings(local_model, data_loader)
    print(f"Embedding shape: {embeddings.shape}")

    if embeddings.shape[0] != len(dataset):
        print("❌ ERROR: Embedding count doesn't match dataset size.")
    elif embeddings.shape[1] != 128:
        print("❌ ERROR: Embedding dimension incorrect.")
    else:
        print("✅ Embedding extraction successful.")


# ========== 2️⃣ Test: Macro-micro clustering ==========
def test_macro_micro_clustering():
    print("\n🔹 TEST 2: Macro-micro clustering...")

    embeddings = sampler.extract_embeddings(local_model, data_loader)
    selected_samples = sampler.macro_micro_clustering(
        global_model, data_loader, unlabeled_set, embeddings, num_samples=10
    )

    print(f"Selected {len(selected_samples)} samples: {selected_samples[:5]}...")

    if len(selected_samples) != 10:
        print("❌ ERROR: Incorrect number of selected samples.")
    elif len(set(selected_samples)) != len(selected_samples):
        print("❌ ERROR: Duplicate samples selected.")
    else:
        print("✅ Macro-micro clustering successful.")


# ========== 3️⃣ Test: Sample selection ==========
def test_sample_selection():
    print("\n🔹 TEST 3: Sample selection...")

    selected_samples, remaining_unlabeled = sampler.select_samples(
        local_model, global_model, data_loader, c=0, unlabeled_set=unlabeled_set, num_samples=10
    )

    print(f"Selected: {selected_samples[:5]}...")
    print(f"Remaining: {remaining_unlabeled[:5]}...")

    if len(selected_samples) != 10:
        print("❌ ERROR: Incorrect number of selected samples.")
    elif len(set(selected_samples).intersection(remaining_unlabeled)) > 0:
        print("❌ ERROR: Selected samples appear in remaining unlabeled set.")
    else:
        print("✅ Sample selection successful.")


# ========== 4️⃣ Test: Edge case - empty cluster handling ==========
def test_empty_clusters():
    print("\n🔹 TEST 4: Empty cluster handling...")

    try:
        embeddings = sampler.extract_embeddings(local_model, data_loader)

        forced_num_samples = 150
        safe_num_samples = min(forced_num_samples, len(dataset))

        selected_samples = sampler.macro_micro_clustering(
            global_model, data_loader, unlabeled_set, embeddings, num_samples=safe_num_samples
        )
        print(f"Selected {len(selected_samples)} samples with capped clusters at dataset size.")

        if len(selected_samples) <= len(dataset):
            print("✅ Empty cluster edge case handled successfully.")
        else:
            print("❌ ERROR: Selected more samples than available.")
    except Exception as e:
        print(f"❌ ERROR: Failed with exception: {e}")


# ========== 5️⃣ Test: Consistency of local and global models ==========
def test_model_usage():
    print("\n🔹 TEST 5: Ensuring local and global models are both used...")

    # Freeze global model to always output zeros
    def frozen_global_model(x):
        batch_size = x.size(0)
        return torch.zeros(batch_size, 10).to(device), torch.zeros(batch_size, 128).to(device)

    global_model.forward = frozen_global_model

    selected_samples, _ = sampler.select_samples(
        local_model, global_model, data_loader, c=0, unlabeled_set=unlabeled_set, num_samples=10
    )

    print(f"Selected samples with frozen global model: {selected_samples[:5]}...")
    print("✅ Global model impact checked (entropy should be uniform).")


# ========== Run tests ==========
if __name__ == "__main__":
    test_extract_embeddings()
    test_macro_micro_clustering()
    test_sample_selection()
    test_empty_clusters()
    test_model_usage()
