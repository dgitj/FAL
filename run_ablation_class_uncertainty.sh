#!/bin/bash
#SBATCH --job-name=ablation_class_uncertainty
#SBATCH --output=logs/ablation_class_uncertainty_%A_%a.out
#SBATCH --error=logs/ablation_class_uncertainty_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# This script runs the ablation study for class-specific uncertainty sampling
# without the two-phase selection or class rebalancing

# Load necessary modules (adjust as needed for your environment)
module load python/3.9
module load cuda/11.3

# Activate environment (adjust as needed)
source /path/to/your/venv/bin/activate

# Set experiment parameters
DATASET="CIFAR10"   # Options: CIFAR10, SVHN, CIFAR100
CLIENTS=10
BUDGET=2500
BASE=5000
ALPHA=0.1          # Dirichlet parameter for non-IID partitioning
TRIALS=3           # Number of trials to run

# Calculate trial ID from SLURM array
TRIAL=$((SLURM_ARRAY_TASK_ID % TRIALS))

# Set seed with offset
SEED=$((44 + TRIAL * 2000000))

# Run experiment with ablation strategy
python main.py \
  --dataset "$DATASET" \
  --strategy "AblationClassUncertainty" \
  --clients "$CLIENTS" \
  --budget "$BUDGET" \
  --base "$BASE" \
  --alpha "$ALPHA" \
  --seed "$SEED" \
  --trial "$TRIAL"

echo "Ablation study completed for trial $TRIAL"
