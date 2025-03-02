import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
from torch.utils.data import DataLoader, TensorDataset

# Import the KAFL Sampler
from query_strategies.kafal import KAFALSampler

class SimpleClassificationModel(nn.Module):
    def __init__(self, input_dim=10, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, num_classes)
    
    def forward(self, x):
        features = F.relu(self.fc1(x))
        scores = self.fc2(features)
        return scores, features

class TestKAFALSampler(unittest.TestCase):
    def setUp(self):
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create sample data
        self.input_dim = 10
        self.num_classes = 5
        self.num_samples = 100
        
        # Generate random input data
        X = torch.randn(self.num_samples, self.input_dim)
        y = torch.randint(0, self.num_classes, (self.num_samples,))
        
        # Create DataLoader
        dataset = TensorDataset(X, y)
        self.unlabeled_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Create sample loss weights
        self.loss_weights = [
            torch.rand(self.num_classes) for _ in range(3)  # Weights for 3 clients
        ]
        
        # Initialize models
        self.client_model = SimpleClassificationModel(
            input_dim=self.input_dim, 
            num_classes=self.num_classes
        )
        self.server_model = SimpleClassificationModel(
            input_dim=self.input_dim, 
            num_classes=self.num_classes
        )
        
        # Create KAFL Sampler
        self.kafal_sampler = KAFALSampler(
            loss_weight_list=self.loss_weights, 
            device='cpu'
        )
    
    def test_initialization(self):
        """Test KAFL Sampler initialization"""
        self.assertIsNotNone(self.kafal_sampler)
        self.assertEqual(self.kafal_sampler.device, 'cpu')
        self.assertEqual(len(self.kafal_sampler.loss_weight_list), 3)
    
    def test_loss_weight_validation(self):
        """Test loss weight list validation"""
        # Test invalid client ID (out of range)
        with self.assertRaises(ValueError):
            self.kafal_sampler.compute_discrepancy(
                self.client_model, 
                self.server_model, 
                self.unlabeled_loader, 
                c=3  # Out of range
            )
    
    def test_discrepancy_computation(self):
        """Test discrepancy computation"""
        # Compute discrepancy for a valid client
        discrepancy = self.kafal_sampler.compute_discrepancy(
            self.client_model, 
            self.server_model, 
            self.unlabeled_loader, 
            c=0
        )
        
        # Check discrepancy tensor properties
        self.assertIsInstance(discrepancy, torch.Tensor)
        self.assertEqual(len(discrepancy), self.num_samples)
        self.assertTrue(torch.all(discrepancy >= 0))
    
    def test_sample_selection(self):
        """Test sample selection process"""
        # Create an unlabeled set of indices
        unlabeled_set = list(range(self.num_samples))
        num_samples_to_select = 20
        
        # Perform sample selection
        selected_samples, remaining_samples = self.kafal_sampler.select_samples(
            self.client_model, 
            self.server_model, 
            self.unlabeled_loader, 
            c=0, 
            unlabeled_set=unlabeled_set, 
            num_samples=num_samples_to_select
        )
        
        # Verify selection properties
        self.assertEqual(len(selected_samples), num_samples_to_select)
        self.assertEqual(len(remaining_samples), self.num_samples - num_samples_to_select)
        
        # Ensure no overlap between selected and remaining samples
        self.assertTrue(len(set(selected_samples) & set(remaining_samples)) == 0)
    
    def test_edge_cases(self):
        """Test edge cases in sample selection"""
        # Try to select more samples than available
        unlabeled_set = list(range(self.num_samples))
        
        # Should not raise an error and limit to available samples
        selected_samples, remaining_samples = self.kafal_sampler.select_samples(
            self.client_model, 
            self.server_model, 
            self.unlabeled_loader, 
            c=0, 
            unlabeled_set=unlabeled_set, 
            num_samples=self.num_samples + 10
        )
        
        self.assertEqual(len(selected_samples), self.num_samples)
        self.assertEqual(len(remaining_samples), 0)
    
    def test_deterministic_behavior(self):
        """Test if sample selection is consistent with fixed seed"""
        # Recreate the setup with fixed seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        # First selection
        first_selected_samples, first_remaining_samples = self.kafal_sampler.select_samples(
            self.client_model, 
            self.server_model, 
            self.unlabeled_loader, 
            c=0, 
            unlabeled_set=list(range(self.num_samples)), 
            num_samples=20
        )
        
        # Reset seeds and rerun
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Second selection
        second_selected_samples, second_remaining_samples = self.kafal_sampler.select_samples(
            self.client_model, 
            self.server_model, 
            self.unlabeled_loader, 
            c=0, 
            unlabeled_set=list(range(self.num_samples)), 
            num_samples=20
        )
        
        # Verify consistency
        self.assertEqual(first_selected_samples, second_selected_samples)
        self.assertEqual(first_remaining_samples, second_remaining_samples)

def run_tests():
    """Run all tests for the KAFL Sampler"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKAFALSampler)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

if __name__ == '__main__':
    run_tests()