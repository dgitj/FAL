import numpy as np
import torch
import torch.nn.functional as F

class AdaptiveDifficultySampler:

    def __init__(self, device="cuda"):
        self.device = device
        self.round = 0
        self.max_rounds = 5
        self.client_performance = {}  # Track performance per client
        
    def calculate_difficulty_spectrum(self, client_model, unlabeled_loader):
        """Calculate multi-dimensional difficulty metrics"""
        entropies = []
        distances = []
        features_list = []
        
        with torch.no_grad():
            for inputs, _ in unlabeled_loader:
                inputs = inputs.to(self.device)
                
                # Get predictions and features
                outputs, features = client_model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                # Calculate entropy (uncertainty)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                
                # Calculate distance from decision boundary
                top2_values, _ = torch.topk(probs, 2, dim=1)
                margin = top2_values[:, 0] - top2_values[:, 1]  # Confidence margin
                
                entropies.append(entropy.cpu())
                distances.append(margin.cpu())
                
                if isinstance(features, list):
                    features = features[-1]  # Use last layer if features is a list
                features_list.append(features.cpu())
        
        # Combine metrics
        entropies = torch.cat(entropies).numpy()
        distances = torch.cat(distances).numpy()
        features = torch.cat(features_list).numpy()
        
        return {
            'entropy': entropies,
            'margin': distances,
            'features': features
        }
    
    def get_target_difficulties(self, client_id):
        """Get target difficulty distribution based on round and client"""
        # Default distribution: equal weight across difficulty spectrum
        easy_weight = 0.33
        medium_weight = 0.34
        hard_weight = 0.33
        
        # Adjust based on round
        round_progress = self.round / self.max_rounds
        easy_weight -= round_progress * 0.23  # Decrease from 0.33 to 0.10
        medium_weight -= round_progress * 0.14  # Decrease from 0.34 to 0.20
        hard_weight += round_progress * 0.37  # Increase from 0.33 to 0.70
        
        # Further adjust based on client performance
        if client_id in self.client_performance:
            performance = self.client_performance[client_id]
            if performance > 0.8:  # Client is doing well
                # Push toward harder samples faster
                easy_weight *= 0.8
                medium_weight *= 0.9
                hard_weight = 1.0 - easy_weight - medium_weight
        
        return {
            'easy': easy_weight,
            'medium': medium_weight,
            'hard': hard_weight
        }
    
    def select_samples(self, client_model, server_model, unlabeled_loader, client_id, unlabeled_set, num_samples, seed=None):
        # Calculate difficulty metrics
        difficulties = self.calculate_difficulty_spectrum(client_model, unlabeled_loader)
        
        # Normalize entropy for easier bucketing
        entropy = difficulties['entropy']
        normalized_entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)
        
        # Create difficulty buckets
        easy_indices = np.where(normalized_entropy < 0.33)[0]
        medium_indices = np.where((normalized_entropy >= 0.33) & (normalized_entropy < 0.66))[0]
        hard_indices = np.where(normalized_entropy >= 0.66)[0]
        
        # Get target difficulties
        targets = self.get_target_difficulties(client_id)
        
        # Calculate samples per bucket
        easy_count = max(1, int(num_samples * targets['easy']))
        medium_count = max(1, int(num_samples * targets['medium']))
        hard_count = num_samples - easy_count - medium_count
        
        # Select samples from each bucket
        selected_indices = []
        
        # Handle case where a bucket might not have enough samples
        if len(easy_indices) < easy_count:
            selected_indices.extend(easy_indices)
            medium_count += (easy_count - len(easy_indices))
        else:
            # Select highest entropy within easy bucket
            bucket_entropies = normalized_entropy[easy_indices]
            top_easy = easy_indices[np.argsort(-bucket_entropies)[:easy_count]]
            selected_indices.extend(top_easy)
            
        if len(medium_indices) < medium_count:
            selected_indices.extend(medium_indices)
            hard_count += (medium_count - len(medium_indices))
        else:
            bucket_entropies = normalized_entropy[medium_indices]
            top_medium = medium_indices[np.argsort(-bucket_entropies)[:medium_count]]
            selected_indices.extend(top_medium)
            
        if len(hard_indices) < hard_count:
            selected_indices.extend(hard_indices)
            # If we still need more, take from any bucket
            remaining = hard_count - len(hard_indices)
            if remaining > 0:
                all_indices = np.arange(len(normalized_entropy))
                unused = np.setdiff1d(all_indices, selected_indices)
                if len(unused) >= remaining:
                    remaining_entropy = normalized_entropy[unused]
                    additional = unused[np.argsort(-remaining_entropy)[:remaining]]
                    selected_indices.extend(additional)
        else:
            bucket_entropies = normalized_entropy[hard_indices]
            top_hard = hard_indices[np.argsort(-bucket_entropies)[:hard_count]]
            selected_indices.extend(top_hard)
        
        # Map to original dataset indices
        selected_samples = [unlabeled_set[i] for i in selected_indices]
        remaining_unlabeled = [idx for idx in unlabeled_set if idx not in selected_samples]
        
        # Update round counter
        self.round += 1
        
        return selected_samples, remaining_unlabeled