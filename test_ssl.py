# test_ssl.py
"""
Simple test script to verify SSL pre-training is working correctly.
Run this to debug SSL issues in isolation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import CIFAR10
import config

# Temporarily set SSL config for testing
config.USE_SSL_PRETRAIN = True
config.SSL_ROUNDS = 5  # Just a few rounds for testing
config.SSL_LOCAL_EPOCHS = 2
config.SSL_BATCH_SIZE = 128
config.CLIENTS = 2  # Test with just 2 clients
config.DATASET = "CIFAR10"

from training.federated_ssl_trainer import FederatedSSLTrainer
from models.ssl_models import create_encoder_cifar

def test_ssl_pretraining():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create raw CIFAR10 dataset (no transforms)
    print("\n1. Loading raw CIFAR10 dataset...")
    dataset = CIFAR10('data/cifar-10-batches-py', train=True, download=True, transform=None)
    print(f"Dataset size: {len(dataset)}")
    
    # Create simple data splits
    print("\n2. Creating data splits...")
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    split_size = len(indices) // config.CLIENTS
    data_splits = []
    for i in range(config.CLIENTS):
        start = i * split_size
        end = start + split_size if i < config.CLIENTS - 1 else len(indices)
        data_splits.append(indices[start:end])
        print(f"Client {i}: {len(data_splits[i])} samples")
    
    # Test SSL trainer
    print("\n3. Testing SSL pre-training...")
    trainer = FederatedSSLTrainer(config, device)
    
    # Run SSL pre-training
    encoder = trainer.federated_ssl_pretrain(
        data_splits=data_splits,
        base_dataset=dataset,
        trial_seed=42
    )
    
    # Test the encoder
    print("\n4. Testing encoder output...")
    encoder.eval()
    
    # Create test transform (same as training would use)
    from torchvision import transforms
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    
    # Test on a few images
    with torch.no_grad():
        features = []
        for i in [0, 100, 200, 300, 400]:
            img, _ = dataset[i]
            img_tensor = test_transform(img).unsqueeze(0).to(device)
            feat = encoder(img_tensor)
            features.append(feat)
            print(f"Image {i} - Feature shape: {feat.shape}, norm: {feat.norm().item():.4f}")
        
        # Check feature diversity
        print("\n5. Checking feature diversity...")
        similarities = []
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                sim = F.cosine_similarity(features[i], features[j])
                similarities.append(sim.item())
                print(f"Similarity between image {i*100} and {j*100}: {sim.item():.4f}")
        
        avg_sim = np.mean(similarities)
        print(f"\nAverage similarity: {avg_sim:.4f}")
        
        if avg_sim > 0.95:
            print("ERROR: Features are too similar - SSL failed!")
        elif avg_sim < 0.3:
            print("SUCCESS: Features are diverse - SSL is working!")
        else:
            print("OK: Features show moderate diversity")
    
    return encoder

if __name__ == "__main__":
    encoder = test_ssl_pretraining()
    print("\nSSL test completed!")
