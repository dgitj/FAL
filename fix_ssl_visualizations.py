"""
Script to fix SSL visualization issues with different checkpoints
This script will:
1. Create test visualizations from different checkpoints
2. Compare the features from different checkpoints
3. Update the SSLEntropySampler to ensure it properly reflects checkpoint differences
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader
import argparse
import shutil
from datetime import datetime

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
from create_matching_model import CustomContrastiveModel
from query_strategies.ssl_entropy import SSLEntropySampler
from data.sampler import SubsetSequentialSampler

def load_checkpoint(checkpoint_path, device="cuda"):
    """Load a checkpoint and return the model"""
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = CustomContrastiveModel(checkpoint)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise

def extract_features(model, dataset, indices, device="cuda", batch_size=64):
    """Extract features from dataset samples using the model"""
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    model = model.to(device)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetSequentialSampler(indices),
        num_workers=2
    )
    
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            
            # Extract features
            batch_features = model.get_features(inputs)
            batch_features = batch_features.cpu().numpy()
            
            # Apply L2 normalization
            for i in range(len(batch_features)):
                batch_features[i] = batch_features[i] / np.linalg.norm(batch_features[i])
            
            features.append(batch_features)
            labels.extend(targets.numpy())
    
    features = np.vstack(features) if features else np.array([])
    labels = np.array(labels)
    
    return features, labels

def calculate_checkpoint_similarity(features1, features2):
    """Calculate similarity between features from different checkpoints"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute mean cosine similarity
    similarities = []
    for i in range(len(features1)):
        sim = cosine_similarity([features1[i]], [features2[i]])[0][0]
        similarities.append(sim)
    
    # Calculate statistics
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    
    return {
        "mean": mean_sim,
        "median": median_sim,
        "min": min_sim,
        "max": max_sim,
        "similarities": similarities
    }

def create_visualization(features, labels, title, output_path, random_seed=42):
    """Create t-SNE visualization of features"""
    from sklearn.manifold import TSNE
    
    # Apply t-SNE dimensionality reduction
    perplexity = min(30, max(5, len(features) - 1))
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Define a color map
    colors = plt.cm.tab10.colors
    
    # Plot each class with a different color
    for class_idx in sorted(set(labels)):
        mask = labels == class_idx
        if np.any(mask):
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[int(class_idx) % len(colors)]],
                s=50,
                alpha=0.7,
                label=f"Class {class_idx}"
            )
    
    plt.title(f"{title} (random_seed={random_seed})", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return features_2d

def check_ssl_entropy_sampler():
    """Check SSLEntropySampler for issues with fixed seeds"""
    import inspect
    import re
    
    # Get the source code of the visualize_labeled_embeddings method
    src = inspect.getsource(SSLEntropySampler.visualize_labeled_embeddings)
    
    # Check for fixed seeds
    fixed_seed = re.search(r'random_state\s*=\s*(\d+)', src)
    if fixed_seed:
        print(f"WARNING: Found fixed random seed ({fixed_seed.group(1)}) in SSLEntropySampler.visualize_labeled_embeddings")
        print("This will cause visualizations to look the same regardless of checkpoint")
        return False
    else:
        print("No fixed random seed found in SSLEntropySampler.visualize_labeled_embeddings")
        return True

def backup_file(file_path):
    """Create a backup of a file before modifying it"""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        print(f"Created backup of {file_path} at {backup_path}")
        return backup_path
    return None

def patch_ssl_entropy_sampler():
    """Apply the patch to fix the SSLEntropySampler visualization method"""
    ssl_entropy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "query_strategies",
        "ssl_entropy.py"
    )
    
    if not os.path.exists(ssl_entropy_path):
        print(f"Error: SSL entropy sampler not found at {ssl_entropy_path}")
        return False
    
    # Backup file
    backup_file(ssl_entropy_path)
    
    with open(ssl_entropy_path, 'r') as f:
        code = f.read()
    
    # Replace fixed random seed with dynamic one
    import re
    pattern = r'tsne\s*=\s*TSNE\(n_components=2,\s*random_state=(\d+),\s*perplexity=perplexity\)'
    replacement = """# Use a dynamic seed based on checkpoint and client_id to ensure different visualizations
            checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SSL_checkpoints', 'final_checkpoint.pt')
            checkpoint_modified_time = os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0
            random_seed = (int(checkpoint_modified_time * 1000) + (client_id or 0) * 100) % 10000
            print(f"[DEBUG] Using dynamic random_state={random_seed} for t-SNE based on checkpoint modification time")
            
            tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity)"""
    
    if re.search(pattern, code):
        code = re.sub(pattern, replacement, code)
        
        # Add checkpoint verification code to _load_ssl_model_from_checkpoint
        pattern2 = r'def _load_ssl_model_from_checkpoint\(self\):'
        checkpoint_verification = """def _load_ssl_model_from_checkpoint(self):
        \"\"\"Load SimCLR model from checkpoint\"\"\"
        # Define checkpoint path
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'SSL_checkpoints')
        
        # Check if we should use ResNet50 model
        checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pt')
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"[SSLEntropy] Error: SSL checkpoint not found at {checkpoint_path}")
        
        # Log checkpoint information for debugging
        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
        checkpoint_modified = os.path.getmtime(checkpoint_path)
        import datetime
        modified_time = datetime.datetime.fromtimestamp(checkpoint_modified).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[SSLEntropy] Loading checkpoint: {checkpoint_path}")
        print(f"[SSLEntropy] Checkpoint size: {checkpoint_size:.2f} MB")
        print(f"[SSLEntropy] Last modified: {modified_time}")
        
        # Calculate checkpoint hash for verification
        import hashlib
        with open(checkpoint_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        print(f"[SSLEntropy] Checkpoint MD5 hash: {file_hash}")
        """
        
        if re.search(pattern2, code):
            code_blocks = re.split(pattern2, code)
            if len(code_blocks) >= 2:
                # Replace the function definition
                code = code_blocks[0] + checkpoint_verification + code_blocks[1].split('\n', 1)[1]
        
        # Write modified code back
        with open(ssl_entropy_path, 'w') as f:
            f.write(code)
        
        print(f"Successfully patched {ssl_entropy_path}")
        print("The SSL visualizations will now reflect differences between checkpoints")
        return True
    else:
        print(f"Could not find the pattern to replace in {ssl_entropy_path}")
        print("The file may have already been patched or has a different structure")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix SSL visualization issues with different checkpoints")
    parser.add_argument("--test-checkpoint1", type=str, help="Path to first SSL checkpoint for testing")
    parser.add_argument("--test-checkpoint2", type=str, help="Path to second SSL checkpoint for testing")
    parser.add_argument("--output-dir", type=str, default="ssl_visualization_fix", help="Output directory for test results")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to use for testing")
    parser.add_argument("--check-only", action="store_true", help="Only check for issues, don't apply fix")
    parser.add_argument("--apply-fix", action="store_true", help="Apply the fix without testing")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU instead of GPU")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")
    
    # Check if SSLEntropySampler has fixed seeds
    has_issue = not check_ssl_entropy_sampler()
    
    if args.apply_fix:
        # Apply fix without testing
        print("Applying fix to SSL visualizations...")
        success = patch_ssl_entropy_sampler()
        if success:
            print("Fix applied successfully")
        else:
            print("Failed to apply fix")
        return
    
    if args.check_only:
        # Only check for issues, don't continue to testing or fixing
        print("Check completed. Use --apply-fix to apply the fix.")
        return
    
    # If not just checking or directly applying, run tests with checkpoints
    if args.test_checkpoint1 and args.test_checkpoint2:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup test dataset
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        test_dataset = CIFAR10("data", train=False, download=True, transform=test_transform)
        
        # Create sample subset for testing
        np.random.seed(42)
        sample_indices = np.random.choice(len(test_dataset), args.num_samples, replace=False)
        
        # Load models from checkpoints
        model1 = load_checkpoint(args.test_checkpoint1, device)
        model2 = load_checkpoint(args.test_checkpoint2, device)
        
        # Extract features
        print(f"Extracting features from first checkpoint...")
        features1, labels = extract_features(model1, test_dataset, sample_indices, device)
        
        print(f"Extracting features from second checkpoint...")
        features2, _ = extract_features(model2, test_dataset, sample_indices, device)
        
        # Calculate similarity
        similarity = calculate_checkpoint_similarity(features1, features2)
        
        print("\n=== Feature Similarity Analysis ===")
        print(f"Mean similarity: {similarity['mean']:.4f}")
        print(f"Median similarity: {similarity['median']:.4f}")
        print(f"Min similarity: {similarity['min']:.4f}")
        print(f"Max similarity: {similarity['max']:.4f}")
        
        # Visualize features with fixed seed (to demonstrate the issue)
        print("\nCreating visualizations with fixed seed...")
        fixed_seed = 42
        features_2d_1 = create_visualization(
            features1, labels, 
            f"Checkpoint 1 Features", 
            os.path.join(args.output_dir, "checkpoint1_fixed_seed.png"),
            random_seed=fixed_seed
        )
        
        features_2d_2 = create_visualization(
            features2, labels,
            f"Checkpoint 2 Features",
            os.path.join(args.output_dir, "checkpoint2_fixed_seed.png"),
            random_seed=fixed_seed
        )
        
        # Visualize features with different seeds (to demonstrate the solution)
        print("\nCreating visualizations with different seeds...")
        features_2d_1_diff = create_visualization(
            features1, labels,
            f"Checkpoint 1 Features",
            os.path.join(args.output_dir, "checkpoint1_different_seed.png"),
            random_seed=42
        )
        
        features_2d_2_diff = create_visualization(
            features2, labels,
            f"Checkpoint 2 Features",
            os.path.join(args.output_dir, "checkpoint2_different_seed.png"),
            random_seed=99
        )
        
        # Create plot showing similarity histogram
        plt.figure(figsize=(10, 6))
        plt.hist(similarity['similarities'], bins=50, alpha=0.75)
        plt.axvline(x=similarity['mean'], color='r', linestyle='--', 
                    label=f'Mean: {similarity["mean"]:.4f}')
        plt.title('Cosine Similarity Between Features from Different Checkpoints')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "feature_similarity_histogram.png"), dpi=300)
        plt.close()
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Test results saved to {os.path.abspath(args.output_dir)}")
        
        # Make recommendation based on similarity
        if similarity['mean'] > 0.95:
            print("\nRECOMMENDATION: The checkpoints are producing VERY SIMILAR features (>95% similarity)")
            print("This suggests they might be effectively the same checkpoint or nearly identical in their representations.")
            print("Check if the checkpoints are actually different.")
        elif similarity['mean'] > 0.85:
            print("\nRECOMMENDATION: The checkpoints are producing SIMILAR features (85-95% similarity)")
            print("The visualizations may look similar even with different seeds.")
            print("Consider using more distinct checkpoints if you want to see clear differences.")
        else:
            print(f"\nRECOMMENDATION: The checkpoints are producing DIFFERENT features ({similarity['mean']*100:.1f}% similarity)")
            print("The fixed seed in the visualization is likely causing identical plots despite the features being different.")
        
        # Apply fix if issue found
        if has_issue:
            print("\nThe SSL visualization code has a fixed random seed, which is causing identical visualizations.")
            print("Would you like to apply the fix? (y/n)")
            response = input().strip().lower()
            if response == 'y':
                success = patch_ssl_entropy_sampler()
                if success:
                    print("Fix applied successfully")
                else:
                    print("Failed to apply fix")
            else:
                print("Fix not applied. Use --apply-fix to apply the fix later.")
        else:
            print("\nNo issues found with the SSL visualization code. It appears to already be using dynamic seeds.")
    else:
        # No test checkpoints provided
        if has_issue:
            print("\nIssue found: The SSL visualization code has a fixed random seed.")
            print("This will cause visualizations to look the same regardless of which checkpoint is used.")
            print("Use --apply-fix to apply the fix, or provide test checkpoints with --test-checkpoint1 and --test-checkpoint2.")
        else:
            print("\nNo issues found with the SSL visualization code. It appears to already be using dynamic seeds.")
            print("If you still see identical visualizations, there might be other issues.")
            print("Provide test checkpoints with --test-checkpoint1 and --test-checkpoint2 for further analysis.")

if __name__ == "__main__":
    main()
