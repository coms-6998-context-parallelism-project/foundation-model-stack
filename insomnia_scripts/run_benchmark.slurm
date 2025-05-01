#!/bin/bash
#
# This script runs the benchmark_attn.py script, potentially in comparison mode.

#SBATCH --account=edu             # Your account name
#SBATCH --job-name=llama_benchmark # Job name appearing in squeue
#SBATCH -N 1                      # Requesting one node
#SBATCH -c 4                      # Requesting 4 CPU cores per task (adjust if needed)
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
echo "Setting up environment..."
REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
MODEL_PATH="/insomnia001/depts/edu/COMSE6998/sg3790/llama-hf"
TOKENIZER_PATH="/insomnia001/depts/edu/COMSE6998/sg3790/llama-hf"

# Navigate to repo
cd "$REPO_DIR" || { echo "Failed to cd to repo $REPO_DIR"; exit 1; }

# Install dependencies (consider doing this once manually or using a virtual env)
echo "Ensuring dependencies are installed..."
pip install --user -e . --quiet
pip install --user safetensors --quiet
pip install --user transformers --quiet
echo "Dependency check complete."

# --- System Info ---
echo "--- System Info ---"
which python
echo "CPU cores requested: $SLURM_CPUS_PER_TASK"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "-------------------"

# --- Run Benchmark Script ---
echo "Starting Python benchmark script..."
# Pass all arguments received by this Slurm script ($@) to the Python script
python scripts/benchmark_attn.py \
    --model_path "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --device_type cuda \
    --dtype fp16 \
    --print_response \
    "$@" # Pass along arguments like --max_new_tokens, --attn_types_to_compare etc.

echo "========================================================"
echo "Benchmark Job finished at $(date)"
echo "========================================================"
