# HybridEntropyKAFAL Strategy

## Overview

HybridEntropyKAFAL is a novel active learning strategy that combines the strengths of entropy-based uncertainty sampling with model discrepancy detection for federated active learning. This hybrid approach is designed to improve selection efficiency by applying different sampling strategies based on class variance characteristics across clients.

## Key Features

- **Dual Sampling Approach**: Uses different strategies for different classes based on their variance characteristics
- **Model Discrepancy**: Leverages both local and global models to identify valuable samples
- **Entropy-First Selection**: Prioritizes high entropy samples across all classes, ensuring informative samples are never discarded due to balancing constraints
- **Variance Analysis**: Identifies the most consistent class across clients to apply specialized sampling

## How It Works

1. **Class Variance Analysis**:
   - Identifies the class with the lowest variance across clients (most consistently distributed)
   - Uses this information to determine which samples to process with each strategy

2. **For the lowest-variance class**:
   - Uses a KAFAL-inspired approach that computes a discrepancy score between the local and global models
   - Prioritizes samples where the local model is confident but disagrees with the global model
   - Optionally applies class weighting to favor underrepresented classes

3. **For all other classes**:
   - Uses standard entropy-based sampling to select the most uncertain samples
   - This maximizes exploration of the model's uncertainty

4. **Global Distribution-Based Selection**:
   - Allocates samples based on the global class distribution
   - Helps maintain the desired class balance across the entire dataset
   - Corrects for class imbalances that might exist in the unlabeled pool

## Usage

To use this strategy in your experiments:

```bash
python main.py --strategy "HybridEntropyKAFAL"
```

For SLURM environments:
```bash
sbatch run_hybrid_strategy.sh
```

## Requirements

The strategy requires:
- Class variance statistics from federated clients
- Access to both local and global models
- Non-IID data distribution (performs best in these scenarios)

## Implementation Details

HybridEntropyKAFAL is implemented in `query_strategies/hybrid_entropy_kafal.py` and integrated into the `strategy_manager.py` framework. The implementation includes:

1. **compute_model_predictions**: Computes predictions from both local and global models
2. **compute_discrepancy_score**: Calculates model discrepancy scores for the low-variance class
3. **select_samples**: Main selection algorithm that applies the hybrid approach

## Experimental Results

This strategy is expected to perform well in non-IID federated settings where clients have specialized knowledge. It combines the strengths of:
- KAFAL's knowledge-aware selection for specialized classes
- Entropy's uncertainty exploration for other classes

This creates a balanced approach that both exploits client specialization and explores model uncertainty.
