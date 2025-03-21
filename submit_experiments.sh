#!/bin/bash
#SBATCH --job-name=fal
#SBATCH --partition=gpu_8
#SBATCH --output=/home/kit/kastel/wu0175/FAL/logs/experiment_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --array=0-6%7

module purge
module load devel/cuda/11.8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fal_env

export PYTHONNOUSERSITE=1

# Define the array of query strategies (adjust based on those defined in config.py)
strategies=(KAFAL Entropy BADGE Random Noise FEAL LOGO)
strategy=${strategies[$SLURM_ARRAY_TASK_ID]}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Strategy: $strategy"
echo "Node: $SLURM_JOB_NODELIST"
echo "Submitted from: $SLURM_SUBMIT_DIR"
date

# Ensure persistent directories exist for logs and results
mkdir -p /home/kit/kastel/wu0175/FAL/logs
mkdir -p /home/kit/kastel/wu0175/FAL/analysis_logs

# Use $TMPDIR (a unique, node-local directory on fast SSD)
echo "Using temporary directory: $TMPDIR"

# Copy your project from persistent storage to $TMPDIR
cp -r /home/kit/kastel/wu0175/FAL $TMPDIR

# Change directory to the copied project in $TMPDIR
cd $TMPDIR/FAL

# Optionally, you could start additional I/O logging here if desired (e.g., using vmstat)

# Record start time
start_time=$(date +%s)

# Run the experiment (your main.py script)
python main.py --strategy "$strategy"

# Record end time and compute duration
end_time=$(date +%s)
duration=$(( end_time - start_time ))
echo "Experiment took $duration seconds."

# Optionally list the contents of the local results directory for debugging
echo "Local results directory content:"
ls -la results

# Copy the analysis_logs folder (which includes your JSON file and other logs) back to persistent storage.
rsync -av analysis_logs /home/kit/kastel/wu0175/FAL/analysis_logs/
# When the job finishes, $TMPDIR will be cleaned automatically.
