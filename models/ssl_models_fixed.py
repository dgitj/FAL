# models/ssl_models_fixed.py
"""
FIXED SSL model components for federated self-supervised pre-training.
Provides encoders and projection heads for SimCLR-style training.
FIXED to match the exact PreAct_ResNet_Cifar architecture output format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.preact_resnet import BasicBlock, preact_resnet8_cifar
from models.preact_resnet_mnist import PreActBlock


class CompatibleSSLEncoder(nn.Module):
    """
    SSL Encoder that EXACTLY matches the PreAct_ResNet_Cifar architecture.
    Extracts the feature part (everything before the final FC layer).
    """
    def __init__(self, full_model):
        super(CompatibleSSLEncoder, self).__init__()
        
        # Extract all layers except the final FC layer
        self.conv1 = full_model.conv1
        self.layer1 = full_model.layer1
        self.layer2 = full_model.layer2
        self.layer3 = full_model.layer3
        self.bn = full_model.bn
        self.relu = full_model.relu
        self.avgpool = full_model.avgpool
        
        # Set output dimension (64 for BasicBlock with expansion=1)
        self.output_dim = 64 * BasicBlock.expansion  # Should be 64
        
    def forward(self, x, return_all_features=False):
        # EXACT same forward pass as PreAct_ResNet_Cifar, but stop before FC
        x = self.conv1(x)
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        
        x = self.bn(out3)
        x = self.relu(x)
        x = self.avgpool(x)
        out4 = x.view(x.size(0), -1)  # This is our feature vector
        
        if return_all_features:
            return out4, [out1, out2, out3, out4]
        else:
            return out4  # Return flattened features (batch_size, 64)


def create_encoder_cifar():
    """Create encoder that EXACTLY matches preact_resnet8_cifar architecture."""
    # Create the full working model (same as in main.py when SSL disabled)
    full_model = preact_resnet8_cifar(num_classes=10)
    
    # Extract encoder part
    encoder = CompatibleSSLEncoder(full_model)
    
    print(f"Created SSL encoder with output_dim: {encoder.output_dim}")
    return encoder


def create_encoder_mnist():
    """Create encoder for MNIST dataset."""
    # For MNIST, we'd need a similar approach
    # For now, return a simple encoder
    from models.preact_resnet_mnist import preact_resnet8_mnist
    full_model = preact_resnet8_mnist(num_classes=10)
    
    # Extract encoder (this might need adjustment for MNIST)
    class MNISTSSLEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            # Extract layers (adjust based on MNIST model structure)
            layers = list(model.children())[:-1]
            self.backbone = nn.Sequential(*layers)
            self.output_dim = 64  # Adjust based on actual MNIST model
            
        def forward(self, x, return_all_features=False):
            x = self.backbone(x)
            if len(x.shape) > 2:
                x = F.adaptive_avg_pool2d(x, 1)
                x = x.view(x.size(0), -1)
            
            if return_all_features:
                # For MNIST, we might not have all intermediate features
                # Return empty list or adjust as needed
                return x, [None, None, None, x]
            else:
                return x
    
    return MNISTSSLEncoder(full_model)


class ProjectionHead(nn.Module):
    """
    Projection head for SSL pre-training.
    Maps encoder features to a lower dimensional space for contrastive learning.
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class PretrainedPreActResNet(nn.Module):
    """
    Combines a pre-trained encoder with a new classification head.
    Used after SSL pre-training to create the final model for active learning.
    EXACTLY matches the original model output format.
    """
    def __init__(self, encoder, num_classes):
        super(PretrainedPreActResNet, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, num_classes)
        
        # Store intermediate feature maps during forward pass
        self.intermediate_features = None
        
    def forward(self, x):
        # Get features AND all intermediate outputs from SSL encoder
        features, intermediate_outputs = self.encoder(x, return_all_features=True)
        
        # Apply classifier
        logits = self.classifier(features)
        
        # Return in the SAME format as original PreAct_ResNet_Cifar
        # Original returns: (logits, [out1, out2, out3, out4])
        return logits, intermediate_outputs


def create_model_with_pretrained_encoder_cifar(encoder, num_classes):
    """
    Create a complete model using pre-trained encoder for CIFAR.
    
    Args:
        encoder: Pre-trained encoder from SSL
        num_classes: Number of output classes
        
    Returns:
        Complete model with encoder + classification head
    """
    return PretrainedPreActResNet(encoder, num_classes)


def create_model_with_pretrained_encoder_mnist(encoder, num_classes):
    """
    Create a complete model using pre-trained encoder for MNIST.
    
    Args:
        encoder: Pre-trained encoder from SSL
        num_classes: Number of output classes
        
    Returns:
        Complete model with encoder + classification head
    """
    return PretrainedPreActResNet(encoder, num_classes)


class SimCLRModel(nn.Module):
    """
    Complete SimCLR model combining encoder and projection head.
    Used during SSL pre-training only.
    """
    def __init__(self, encoder, projection_head):
        super(SimCLRModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        
    def forward(self, x):
        # During SSL training, we don't need all features
        features = self.encoder(x, return_all_features=False)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=1)


# Test function to verify compatibility
def test_ssl_compatibility():
    """Test that the SSL encoder produces compatible outputs."""
    print("=" * 50)
    print("TESTING SSL ENCODER COMPATIBILITY")
    print("=" * 50)
    
    # Create original model
    original_model = preact_resnet8_cifar(num_classes=10)
    
    # Create SSL encoder
    ssl_encoder = create_encoder_cifar()
    
    # Test input
    test_input = torch.randn(2, 3, 32, 32)
    
    with torch.no_grad():
        # Test original model
        orig_logits, orig_features = original_model(test_input)
        print(f"Original model output: logits {orig_logits.shape}, features {len(orig_features)} items")
        for i, feat in enumerate(orig_features):
            if feat is not None:
                print(f"  Feature {i}: {feat.shape}")
        
        # Test SSL encoder (without all features)
        ssl_features = ssl_encoder(test_input, return_all_features=False)
        print(f"\nSSL encoder output (training mode): {ssl_features.shape}")
        
        # Test SSL encoder (with all features)
        ssl_features_final, ssl_all_features = ssl_encoder(test_input, return_all_features=True)
        print(f"\nSSL encoder output (inference mode):")
        print(f"  Final features: {ssl_features_final.shape}")
        print(f"  All features: {len(ssl_all_features)} items")
        for i, feat in enumerate(ssl_all_features):
            if feat is not None:
                print(f"    Feature {i}: {feat.shape}")
        
        # Test SSL + classifier
        ssl_model = create_model_with_pretrained_encoder_cifar(ssl_encoder, 10)
        ssl_logits, ssl_features_list = ssl_model(test_input)
        print(f"\nSSL model output: logits {ssl_logits.shape}, features {len(ssl_features_list)} items")
        
        # Check compatibility
        features_match = True
        if len(orig_features) != len(ssl_features_list):
            features_match = False
        else:
            for i, (orig_f, ssl_f) in enumerate(zip(orig_features, ssl_features_list)):
                if orig_f is not None and ssl_f is not None:
                    if orig_f.shape != ssl_f.shape:
                        features_match = False
                        print(f"  Feature {i} mismatch: orig {orig_f.shape} vs ssl {ssl_f.shape}")
        
        if orig_logits.shape == ssl_logits.shape and features_match:
            print("\n✅ SUCCESS: SSL encoder is fully compatible!")
            return True
        else:
            print("\n❌ FAILED: Output format doesn't match!")
            if orig_logits.shape != ssl_logits.shape:
                print(f"  Logits mismatch: {orig_logits.shape} vs {ssl_logits.shape}")
            if not features_match:
                print(f"  Features list mismatch")
            return False


if __name__ == "__main__":
    test_ssl_compatibility()
