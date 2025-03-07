#!/bin/bash
#SBATCH --job-name=fal-array
#SBATCH --partition=gpu_4_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --array=0-1%2
#SBATCH --output=fal_array_%A_%a.out
#SBATCH --error=fal_array_%A_%a.err

trap "kill $IO_MONITOR_PID" EXIT
sleep 5

# Clean environment
module purge
module load devel/cuda/11.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fal_env

export PYTHONNOUSERSITE=1

echo "============================================================"
echo "Job running on node: $(hostname)"
echo "Starting at: $(date)"
echo "============================================================"

nvidia-smi

# Copy necessary files to $TMPDIR (fast local SSD)
echo "Copying job files to \$TMPDIR..."
cp -r $SLURM_SUBMIT_DIR/* $TMPDIR/

# Set DATA_ROOT to use the dataset in $TMPDIR
export DATA_ROOT=$TMPDIR/data

# Move into working directory
cd $TMPDIR

# Start I/O monitoring in the background
IO_LOG="io_monitor_${SLURM_ARRAY_TASK_ID}.log"
(
  while true; do
    date >> $IO_LOG
    for pid in $(pgrep -u $USER python); do
      echo "PID $pid:" >> $IO_LOG
      cat /proc/$pid/io >> $IO_LOG
    done
    sleep 10
  done
) &
IO_MONITOR_PID=$!

# Define experiment parameters
STRATEGIES=("KAFAL" "LOGO")
CLIENT_COUNTS=(40)

STRATEGY_IDX=$(($SLURM_ARRAY_TASK_ID / ${#CLIENT_COUNTS[@]}))
CLIENT_IDX=$(($SLURM_ARRAY_TASK_ID % ${#CLIENT_COUNTS[@]}))

if [ $STRATEGY_IDX -ge ${#STRATEGIES[@]} ]; then
    echo "Error: Strategy index $STRATEGY_IDX out of bounds"
    exit 1
fi

STRATEGY=${STRATEGIES[$STRATEGY_IDX]}
CLIENTS=${CLIENT_COUNTS[$CLIENT_IDX]}
EXPERIMENT_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOG_FILE="experiment_${EXPERIMENT_ID}.log"


echo "============================================================"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Running strategy: $STRATEGY with $CLIENTS clients"
echo "============================================================"

echo "Starting experiment..."
python main_multiprocessing.py \
  --strategy $STRATEGY \
  --clients $CLIENTS \
  --epochs 40 \
  --communication_rounds 50 >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Experiment failed with exit code $EXIT_CODE"
fi

# Stop I/O monitoring
kill $IO_MONITOR_PID

# Copy results back to $HOME only at the end
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${STRATEGY}/${CLIENTS}_clients/${EXPERIMENT_ID}"
mkdir -p "$RESULTS_DIR"

# Copy experiment results
if [ -d "experiments" ]; then
    cp -r experiments/* "$RESULTS_DIR"/ || { echo "Failed to copy results"; exit 1; }
    echo "Results copied to $RESULTS_DIR"
else
    echo "Warning: No experiments directory found. No results copied."
fi


# Copy experiment log and I/O log
cp "$LOG_FILE" "$RESULTS_DIR/" || echo "Warning: Failed to copy experiment log."
cp "$IO_LOG" "$RESULTS_DIR/" || echo "Warning: Failed to copy I/O log."

echo "============================================================"
echo "Experiment completed at: $(date)"
echo "Results stored in: $RESULTS_DIR"
echo "Exit code: $EXIT_CODE"
echo "============================================================"
