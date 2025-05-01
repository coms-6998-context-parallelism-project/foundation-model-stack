#!/bin/bash
#
# Slurm script to run testing/run_comparison.py

#SBATCH --account=edu             # Your account name
#SBATCH --job-name=fms_compare    # Job name appearing in squeue
#SBATCH -N 1                      # Requesting one node
#SBATCH --ntasks-per-node=1       # Run one task per node (torchrun handles nproc_per_node)
#SBATCH --time=01:00:00           # Max job time (adjust based on expected run time)
#SBATCH --mem=80G                 # Memory request (adjust based on model size, might need more)
#SBATCH --gres=gpu:2              # Requesting 2 GPUs (for potential ring usage)
#SBATCH --output=comparison_%j.out # Standard output file (%j expands to job ID)

echo "========================================================"
echo "Starting Comparison Job on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Arguments passed to Python script: $@"
echo "========================================================"

# --- Environment Setup ---
# Load necessary modules (adjust based on your cluster environment)
# module load anaconda3/2023.09
# module load cuda/11.8 # Or the CUDA version required by your PyTorch install

# Activate your conda environment
# source activate your_conda_env_name # Replace with your environment name

# --- Set Python Path ---
# Ensure the FMS library is in the Python path
# Adjust this path if your repository structure is different
REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack" # Adjust base path if needed
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

# --- Ensure Dependencies ---
echo "[INFO] Checking Python and pip paths..."
which python
which pip
echo "[INFO] Attempting to install/verify sentencepiece..."
# Use -v for verbose output, check exit code
pip install --user -v sentencepiece || { echo "[ERROR] Failed to install sentencepiece"; exit 1; }
echo "[INFO] Verifying sentencepiece installation..."
python -m pip show sentencepiece || echo "[WARN] sentencepiece verification failed, but proceeding anyway..."

# --- Run the Comparison Script ---
# Use torchrun for potential distributed execution (like ring attention).
# nproc_per_node=2 assumes ring attention needs 2 GPUs. Adjust if necessary.
# The python script run_comparison.py handles loading models with different attention types.
echo "Running comparison script..."
torchrun --nproc_per_node=2 \
    "${REPO_DIR}/testing/run_comparison.py" \
    --device_type cuda \
    --dtype fp16 \
    "$@" # Pass along all arguments from submit_comparison.sh

EXIT_CODE=$?
echo "========================================================"
echo "Comparison Job finished at $(date) with exit code $EXIT_CODE"
echo "========================================================"
exit $EXIT_CODE

