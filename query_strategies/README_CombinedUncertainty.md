# Combined Local-Global Uncertainty Strategy

## Overview

The `HybridEntropyKAFALEntropyFirst` strategy now uses a combined local-global uncertainty approach. This strategy combines the uncertainty (entropy) from both local and global models to select the most informative samples while maintaining gentle class balancing.

## Key Innovations

1. **Combined Uncertainty Scoring**:
   - Computes entropy uncertainty from both local and global models
   - Combines them (with equal weighting by default) into a single score
   - Uses this combined uncertainty score for sample selection in all classes

2. **Two-Phase Selection**:
   - **First Phase (30%)**: Selects the top 30% of samples with the highest combined uncertainty, regardless of class
   - **Second Phase (70%)**: Applies gentle class balancing based on global distribution targets

3. **Consistent Selection Criterion**:
   - Uses the same combined uncertainty metric for both low-variance and other classes
   - Replaces the discrepancy-based selection that was previously used for the low-variance class
   - Provides a more unified selection criterion while still accounting for global class distribution

## Advantages

1. **More Comprehensive Uncertainty**:
   - Captures uncertainty from both model perspectives
   - Less susceptible to any single model's biases or peculiarities
   - Identifies samples that might be uncertain to either the local or global model

2. **Unified Selection Criterion**:
   - Uses same method across all classes for more consistent selection
   - Simplifies the approach while maintaining effectiveness
   - Easier to understand and analyze

3. **Entropy-First Approach**:
   - Guarantees that high-uncertainty samples are always selected
   - Less rigid than pure class balancing
   - Better exploration of the uncertainty space

## Usage

```bash
python main.py --strategy "HybridEntropyKAFALEntropyFirst"
```

## Implementation Details

1. **Uncertainty Calculation**:
   ```python
   # Combine local and global uncertainty with equal weighting
   combined_entropy = 0.5 * local_entropy + 0.5 * global_entropy
   ```

2. **Phase 1 (Entropy-First)**:
   - Selects top 30% samples based purely on combined uncertainty

3. **Phase 2 (Gentle Balancing)**:
   - Calculates how many samples are still needed for each class based on global distribution
   - Selects remaining samples within each class based on combined uncertainty

4. **Statistics**:
   - Outputs diagnostic information showing average local and global entropy for selected samples
   - Helps verify that selected samples have high uncertainty from both perspectives

## Comparison to Other Approaches

- **Original KAFAL**: Uses a complex loss-weight mechanism
- **Entropy-Only**: Uses only local model uncertainty
- **Discrepancy-Based**: Looks at disagreement between models
- **Combined Uncertainty**: Gets the best of both worlds - consideration of both local and global uncertainty
