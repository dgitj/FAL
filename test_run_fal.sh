#!/bin/bash
#SBATCH --job-name=fal-test
#SBATCH --partition=dev_gpu_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:20:00
#SBATCH --mem=94G
#SBATCH --array=0-3%4
#SBATCH --output=fal_test_%A_%a.out
#SBATCH --error=fal_test_%A_%a.err

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Clean environment to avoid conflicts
module purge

# Load CUDA
module load devel/cuda/11.8

# Activate Miniconda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fal_env

# Run quick test to verify CUDA is available to PyTorch
echo "PyTorch CUDA Check:"
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version compiled with PyTorch:', torch.version.cuda); print('Device count:', torch.cuda.device_count()); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

export PYTHONNOUSERSITE=1

nvidia-smi


# Define exactly 4 experiment configurations
STRATEGIES=("Entropy" "BADGE" "Random")
CLIENT_COUNTS=( 10 10 10)

# Map SLURM_ARRAY_TASK_ID directly to strategy and clients
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
CLIENTS=${CLIENT_COUNTS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "============================================================"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Running strategy: $STRATEGY with $CLIENTS clients"
echo "Starting at: $(date)"
echo "Node: $(hostname)"
echo "============================================================"
echo "GPU information:"
nvidia-smi

echo "Starting experiment..."
python main_multiprocessing.py \
  --strategy $STRATEGY \
  --clients $CLIENTS \
  --epochs 2 \
  --communication_rounds 2 \
  --cycles 2


EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Experiment failed with exit code $EXIT_CODE"
fi


RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${STRATEGY}/${CLIENTS}_clients/${EXPERIMENT_ID}"
mkdir -p "$RESULTS_DIR" || { echo "Failed to create results directory"; exit 1; }

if [ -d "experiments" ]; then
    cp -r experiments/* "$RESULTS_DIR"/ || { echo "Failed to copy results"; exit 1; }
    echo "Results copied to $RESULTS_DIR"
else
    echo "Warning: No experiments directory found"
    if ls results_* 1> /dev/null 2>&1; then
        cp -r results_* "$RESULTS_DIR"/ || { echo "Failed to copy results"; exit 1; }
        echo "Results copied from current directory to $RESULTS_DIR"
    fi
fi

echo "============================================================"
echo "Experiment completed at: $(date)"
echo "Results stored in: $RESULTS_DIR"
echo "Exit code: $EXIT_CODE"
echo "============================================================"
