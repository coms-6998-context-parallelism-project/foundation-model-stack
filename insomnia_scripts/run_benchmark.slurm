#!/bin/bash
#
# This script runs the benchmark_attn.py script, potentially in comparison mode.

#SBATCH --account=edu             # Your account name
#SBATCH --job-name=llama_benchmark # Job name appearing in squeue
#SBATCH -N 1                      # Requesting one node
#SBATCH --ntasks-per-node=1       # Run one task per node
#SBATCH --time=00:30:00           # Max job time (adjust based on expected benchmark duration)
#SBATCH --mem=40G                 # Memory request (adjust if needed)
#SBATCH --gres=gpu:2              # Requesting 2 GPUs (needed for nproc_per_node=2 for ring)
#SBATCH --output=benchmark_compare_%j.out # Standard output file (%j expands to job ID) - Changed name

echo "========================================================"
echo "Starting Benchmark Job on $(hostname) at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Arguments passed to Python script: $@" # No longer shifting args here
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
export PYTHONPATH="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack:$PYTHONPATH" # Adjust base path if needed

# --- Run the Benchmark Script ---
# Use torchrun for potential distributed execution (like ring attention)
# nproc_per_node=2 is often needed for ring attention to utilize 2 GPUs.
# Adjust nproc_per_node based on the requirements of the attention mechanisms being tested.
# The python script itself now handles loading models with different attention types.
echo "Running benchmark script..."
torchrun --nproc_per_node=2 \
    /insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack/scripts/benchmark_attn.py \
    --architecture llama \
    --variant 7b \
    --model_path /insomnia001/depts/edu/COMSE6998/shared/models/hf/meta-llama/Llama-2-7b-hf \
    --tokenizer /insomnia001/depts/edu/COMSE6998/shared/models/hf/meta-llama/Llama-2-7b-hf/tokenizer.model \
    --device_type cuda \
    --dtype fp16 \
    --print_response \
    "$@" # Pass along arguments like --max_new_tokens, --attn_types_to_compare etc.

echo "========================================================"
echo "Benchmark Job finished at $(date)"
echo "========================================================"
