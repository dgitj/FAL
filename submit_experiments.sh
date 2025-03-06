#!/bin/bash
#SBATCH --job-name=fal-array
#SBATCH --partition=gpu_4_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --array=0-20%8
#SBATCH --output=fal_array_%A_%a.out
#SBATCH --error=fal_array_%A_%a.err

# Load necessary modules
module load devel/python/3.10.0_gnu_11.1
module load cuda/12.1

# Activate virtual environment
source ~/venvs/decal_env/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Define experiment parameters
STRATEGIES=("KAFAL" "Entropy" "BADGE" "Random" "FEAL" "LOGO" "Noise")
CLIENT_COUNTS=(10 20 40)

# Calculate which strategy and client count to use based on array index
STRATEGY_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
CLIENT_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

# Verify indices are within range
if [ $STRATEGY_IDX -ge ${#STRATEGIES[@]} ]; then
    echo "Error: Strategy index $STRATEGY_IDX out of bounds"
    exit 1
fi
if [ $CLIENT_IDX -ge ${#CLIENT_COUNTS[@]} ]; then
    echo "Error: Client count index $CLIENT_IDX out of bounds"
    exit 1
fi

# Get the actual strategy and client count
STRATEGY=${STRATEGIES[$STRATEGY_IDX]}
CLIENTS=${CLIENT_COUNTS[$CLIENT_IDX]}

# Create a unique experiment ID
EXPERIMENT_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Print experiment details
echo "============================================================"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Running strategy: $STRATEGY with $CLIENTS clients"
echo "Starting at: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# Check GPU status
echo "GPU information:"
nvidia-smi

# Run the experiment
# Note: Fixed parameter names to match your script's expectations
echo "Starting experiment..."
python main_multiprocessing.py \
  --strategies $STRATEGY \
  --clients $CLIENTS \
  --epoch 40 \
  --communication 50 \


# Check if the experiment succeeded
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Experiment failed with exit code $EXIT_CODE"
fi

# Create and populate results directory with unique ID
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${STRATEGY}_${CLIENTS}_${EXPERIMENT_ID}"
mkdir -p $RESULTS_DIR || { echo "Failed to create results directory"; exit 1; }

# Look for result directories and copy them
if [ -d "experiments" ]; then
    cp -r experiments/* $RESULTS_DIR/ || { echo "Failed to copy results"; exit 1; }
    echo "Results copied to $RESULTS_DIR"
else
    echo "Warning: No experiments directory found"
    # Check for results in current directory
    if ls results_* 1> /dev/null 2>&1; then
        cp -r results_* $RESULTS_DIR/ || { echo "Failed to copy results"; exit 1; }
        echo "Results copied from current directory to $RESULTS_DIR"
    fi
fi

# Record completion time
echo "============================================================"
echo "Experiment completed at: $(date)"
echo "Results stored in: $RESULTS_DIR"
echo "Exit code: $EXIT_CODE"
echo "============================================================"