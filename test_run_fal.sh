#!/bin/bash
#SBATCH --job-name=fal-test
#SBATCH --partition=gpu_4_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=fal_test_%j.out

# Load modules
module load devel/python/3.10.0_gnu_11.1
module load cuda/12.1

# Activate environment
source ~/venvs/decal_env/bin/activate

# Experiment parameters
STRATEGY="Random"
CLIENT_COUNT=10

# Prepare experiment directory
EXP_DIR="${SLURM_SUBMIT_DIR}/test_experiment_${STRATEGY}_${CLIENT_COUNT}"
mkdir -p $EXP_DIR
cd $EXP_DIR

# Copy necessary files
cp -r ${SLURM_SUBMIT_DIR}/models ${SLURM_SUBMIT_DIR}/query_strategies ${SLURM_SUBMIT_DIR}/data ${SLURM_SUBMIT_DIR}/*.py .

# Run the experiment
LOG_FILE="experiment_test.log"
echo "Running TEST ${STRATEGY} with ${CLIENT_COUNT} clients..." > $LOG_FILE
python ${SLURM_SUBMIT_DIR}/main_multiprocessing.py \
  --strategies $STRATEGY \
  --clients $CLIENT_COUNT \
  --processes 1 \
  --epochs 10 \
  --communication_rounds 10 >> $LOG_FILE 2>&1

echo "Test experiment finished." >> $LOG_FILE
