# Entropy-First Hybrid Strategy

## Overview

The `HybridEntropyKAFALEntropyFirst` strategy is an improved version of the `HybridEntropyKAFAL` approach that prioritizes high-entropy samples while still maintaining some class balancing.

## Approach

This strategy combines the best aspects of uncertainty-based selection with intelligent class balancing:

1. **Two-Phase Selection**:
   - **First Phase (30%)**: Selects the top 30% of samples purely based on entropy, regardless of class
   - **Second Phase (70%)**: Applies gentle class balancing for the remaining samples

2. **Class-Specific Selection Methods**:
   - For the lowest-variance class: Uses model discrepancy approach
   - For all other classes: Uses entropy-based sampling

3. **Adaptive Class Allocation**:
   - Accounts for what's already been selected in the first phase
   - Adjusts second phase allocation to achieve global distribution targets
   - Falls back to entropy if targets can't be met

## Advantages

1. **Preserves High-Entropy Samples**: Ensures no valuable high-uncertainty samples are discarded due to rigid balancing requirements
2. **Maintains Gentle Balancing**: Still promotes class balance over multiple rounds, just less aggressively
3. **Best of Both Worlds**: Combines exploration (entropy) with representation balance

## Usage

```bash
python main.py --strategy "HybridEntropyKAFALEntropyFirst"
```

For SLURM environments:
```bash
sbatch run_entropy_first.sh
```

## Comparison to Standard Approach

The key difference from the standard `HybridEntropyKAFAL` strategy:

- **Standard**: Applies class balancing to the entire selection budget
- **Entropy-First**: Reserves 30% of budget for pure entropy-based selection before balancing

This makes the Entropy-First approach more focused on finding the most informative samples while still maintaining some degree of class balancing.
