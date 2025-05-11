# Class-Differentiated Uncertainty Strategy

## Overview

The `HybridEntropyKAFALClassDifferentiated` strategy uses different uncertainty measurement approaches for different classes based on their variance across clients. This approach recognizes that classes with different distribution patterns across clients may benefit from different selection criteria.

## Key Features

1. **Class-Specific Uncertainty Measurement**:
   - **Low-Variance Classes** (the two with lowest variance): Uses combined local-global entropy
   - **High-Variance Classes** (all others): Uses only local entropy

2. **Two-Phase Selection**:
   - **First Phase (30%)**: Selects top samples based on local entropy regardless of class
   - **Second Phase (70%)**: Applies gentle class balancing using class-specific uncertainty metrics

3. **Comprehensive Analysis**:
   - Provides detailed statistics on selected samples from each class group
   - Allows observation of how different uncertainty metrics perform for different classes

## How It Works

1. **Variance Analysis**:
   - Identifies the two classes with the lowest variance across clients
   - These are classes that are most evenly distributed across the federation

2. **First Phase: Pure Entropy Selection**:
   - Uses local entropy to select the top 30% of samples regardless of class
   - This ensures high-information samples are never missed

3. **Second Phase: Class-Differentiated Selection**:
   - For each class, calculates how many more samples are needed based on global distribution
   - For the two low-variance classes:
     - Uses combined local-global uncertainty
     - Weighted equally (0.5 * local + 0.5 * global)
   - For high-variance classes:
     - Uses only local entropy
     - This focuses on samples uncertain to the local model

4. **Global Distribution Alignment**:
   - Gradually moves the overall distribution toward the global target
   - Accounts for what was already selected in the first phase

## Advantages

1. **Optimized Selection for Each Class Type**:
   - Low-variance classes benefit from considering both models' perspectives
   - High-variance classes focus on local uncertainty where the client has specialized knowledge

2. **Preserves High-Information Samples**:
   - First phase ensures no valuable samples are discarded due to class balancing

3. **Detailed Diagnostic Information**:
   - Provides statistics comparing uncertainty metrics across class groups
   - Helps understand the selection behavior

## Usage

```bash
python main.py --strategy "HybridEntropyKAFALClassDifferentiated"
```

For SLURM environments:
```bash
sbatch run_class_differentiated.sh
```

## Theory and Motivation

The motivation for this differentiated approach is based on the understanding that:

1. **Low-variance classes** (evenly distributed across clients) benefit from considering the global model's perspective, as these classes are represented similarly across the federation.

2. **High-variance classes** (unevenly distributed) benefit more from the local model's uncertainty, as the client may have specialized knowledge about these classes that the global model lacks.

By applying the appropriate uncertainty metric to each class type, we can make more informed selections that respect both the client's specialized knowledge and the global perspective.
