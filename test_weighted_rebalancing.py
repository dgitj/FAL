"""
Test script for the weighted rebalancing in HybridEntropyKAFAL strategy.
This script demonstrates how the weighted rebalancing affects sample selection.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from query_strategies.hybrid_entropy_kafal import HybridEntropyKAFALSampler

class SimpleCNN(torch.nn.Module):
    """Simple CNN model for testing."""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_mock_data(device):
    """Create mock data for testing."""
    # Create mock data with uneven class distribution
    # 10 classes with varying representation in unlabeled pool
    num_classes = 10
    num_samples = 1000
    
    # Generate inputs (random images)
    inputs = torch.randn(num_samples, 3, 32, 32).to(device)
    
    # Create labels with uneven distribution
    # Class 0: 5%, Class 1: 10%, Class 2: 15%, etc.
    class_proportions = np.array([0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    class_counts = (class_proportions * num_samples).astype(int)
    labels = []
    for i in range(num_classes):
        labels.extend([i] * class_counts[i])
    
    # Ensure we have exactly num_samples
    while len(labels) < num_samples:
        labels.append(np.random.randint(0, num_classes))
    labels = labels[:num_samples]
    
    # Convert to tensor
    labels = torch.tensor(labels).long().to(device)
    
    # Create DataLoader
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Create unlabeled set indices
    unlabeled_indices = list(range(num_samples))
    
    return dataloader, unlabeled_indices, labels

def test_weighted_rebalancing():
    """Test the weighted rebalancing functionality."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create mock data
    unlabeled_loader, unlabeled_set, all_labels = create_mock_data(device)
    
    # Create mock models
    local_model = SimpleCNN().to(device)
    global_model = SimpleCNN().to(device)
    
    # Create the sampler
    sampler = HybridEntropyKAFALSampler(device=device)
    
    # Create mock class variance statistics
    # Let's say class 2 has the lowest variance
    class_variance_stats = {
        'class_stats': {
            '0': {'variance': 0.05},
            '1': {'variance': 0.04},
            '2': {'variance': 0.02},  # Lowest variance
            '3': {'variance': 0.03},
            '4': {'variance': 0.07},
            '5': {'variance': 0.06},
            '6': {'variance': 0.08},
            '7': {'variance': 0.09},
            '8': {'variance': 0.10},
            '9': {'variance': 0.11}
        }
    }
    
    # Create mock global class distribution
    # Let's say we want a balanced distribution (10% each)
    global_class_distribution = {
        '0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1,
        '5': 0.1, '6': 0.1, '7': 0.1, '8': 0.1, '9': 0.1
    }
    
    # Test with three scenarios for current labeled set distribution
    
    # Scenario 1: Class 2 is underrepresented (5%)
    print("\n===== SCENARIO 1: LOW-VARIANCE CLASS IS UNDERREPRESENTED =====")
    labeled_set_classes_1 = np.array([0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7])
    print(f"Current labeled set has {np.sum(labeled_set_classes_1 == 2)} samples of class 2 out of {len(labeled_set_classes_1)}")
    
    # Select samples
    selected_1, remaining_1 = sampler.select_samples(
        local_model, global_model, unlabeled_loader, 0, unlabeled_set,
        num_samples=50, labeled_set_classes=labeled_set_classes_1,
        global_class_distribution=global_class_distribution,
        class_variance_stats=class_variance_stats
    )
    
    # Scenario 2: Class 2 is properly represented (10%)
    print("\n===== SCENARIO 2: LOW-VARIANCE CLASS IS PROPERLY REPRESENTED =====")
    labeled_set_classes_2 = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
    print(f"Current labeled set has {np.sum(labeled_set_classes_2 == 2)} samples of class 2 out of {len(labeled_set_classes_2)}")
    
    # Select samples
    selected_2, remaining_2 = sampler.select_samples(
        local_model, global_model, unlabeled_loader, 0, unlabeled_set,
        num_samples=50, labeled_set_classes=labeled_set_classes_2,
        global_class_distribution=global_class_distribution,
        class_variance_stats=class_variance_stats
    )
    
    # Scenario 3: Class 2 is overrepresented (20%)
    print("\n===== SCENARIO 3: LOW-VARIANCE CLASS IS OVERREPRESENTED =====")
    labeled_set_classes_3 = np.array([0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 4, 5, 6, 7])
    print(f"Current labeled set has {np.sum(labeled_set_classes_3 == 2)} samples of class 2 out of {len(labeled_set_classes_3)}")
    
    # Select samples
    selected_3, remaining_3 = sampler.select_samples(
        local_model, global_model, unlabeled_loader, 0, unlabeled_set,
        num_samples=50, labeled_set_classes=labeled_set_classes_3,
        global_class_distribution=global_class_distribution,
        class_variance_stats=class_variance_stats
    )
    
    # Compare results
    print("\n===== COMPARISON OF ALLOCATIONS =====")
    print(f"Scenario 1 (Underrepresented): Selected {len(selected_1)} samples")
    print(f"Scenario 2 (Properly Represented): Selected {len(selected_2)} samples")
    print(f"Scenario 3 (Overrepresented): Selected {len(selected_3)} samples")

if __name__ == "__main__":
    test_weighted_rebalancing()
