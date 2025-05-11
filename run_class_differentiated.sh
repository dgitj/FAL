#!/bin/bash
#SBATCH --job-name=class_diff
#SBATCH --partition=gpu_h100
#SBATCH --output=/home/ka/ka_kastel/ka_wu0175/FAL/logs/class_diff_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --array=0-0

module purge
module load devel/cuda/11.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fal

export PYTHONNOUSERSITE=1

# Using the new class-differentiated strategy
strategies=(HybridEntropyKAFALClassDifferentiated)  
strategy=${strategies[$SLURM_ARRAY_TASK_ID]}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Strategy: $strategy"
echo "Node: $SLURM_JOB_NODELIST"
echo "Submitted from: $SLURM_SUBMIT_DIR"
date

# Ensure persistent directories exist for logs and results
mkdir -p /home/ka/ka_kastel/ka_wu0175/FAL/logs
mkdir -p /home/ka/ka_kastel/ka_wu0175/FAL/analysis_logs

# Use $TMPDIR (a unique, node-local directory on fast SSD)
echo "Using temporary directory: $TMPDIR"

# Copy your project from persistent storage to $TMPDIR
cp -r /home/ka/ka_kastel/ka_wu0175/FAL $TMPDIR/

# Also copy the datasets explicitly to ensure they're properly available
mkdir -p $TMPDIR/FAL/data
echo "Copying datasets to temporary directory..."
cp -r /home/ka/ka_kastel/ka_wu0175/FAL/data/cifar-10-batches-py $TMPDIR/FAL/data/
cp -r /home/ka/ka_kastel/ka_wu0175/FAL/data/cifar-100-python $TMPDIR/FAL/data/
cp -r /home/ka/ka_kastel/ka_wu0175/FAL/data/svhn $TMPDIR/FAL/data/

# Change directory to the copied project in $TMPDIR
cd $TMPDIR/FAL

# Add debugging for dataset availability
ls -la data/
echo "CIFAR-10 directory content:"
ls -la data/cifar-10-batches-py/

# Record start time
start_time=$(date +%s)

# Run the experiment with unbuffered output for better debugging
python -u main.py --strategy "$strategy" \
    --cycles 2 \
    --clients 10 \
    --base 5000 \
    --budget 2500 \
    --alpha 0.1 \
    --dataset CIFAR10

# Record end time and compute duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Experiment took $duration seconds."

# List the contents of the local results directory for debugging
echo "Local results directory content:"
ls -la results

# Copy the analysis_logs folder back to persistent storage
rsync -av --exclude analysis_logs/ analysis_logs/ /home/ka/ka_kastel/ka_wu0175/FAL/analysis_logs/

# When the job finishes, $TMPDIR will be cleaned automatically.
