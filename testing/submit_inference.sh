#!/bin/bash
# Ring Attention Inference submit script:
# Submits a single Slurm job (Insomnia/GPU) or runs a local process (CUDA only)
# to execute scripts/inference.py with Ring Attention strategy.

# --- Environment Detection ---
INSOMNIA_REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
if [[ -d "$INSOMNIA_REPO_DIR" ]]; then
  RUN_LOCATION="insomnia"
  echo "[INFO] Detected Insomnia environment. Will use Slurm for GPU execution."
else
  RUN_LOCATION="local"
  echo "[INFO] Detected Local environment. Will use Python directly for CPU execution."
fi

# --- Base Paths ---
LOCAL_REPO_DIR="/Users/sadigulcelik/Documents/CompSci/HPML-2025-Spring/FMSwrapper/foundation-model-stack" # Adjust if needed

# --- Default Paths (relative to repo root) ---
DEFAULT_MODEL_REL_PATH="llama-hf"
DEFAULT_TOKENIZER_REL_PATH="llama-hf/tokenizer.model"

# --- Set Current Paths ---
if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  CURRENT_REPO_DIR="$INSOMNIA_REPO_DIR"
  # Assuming the slurm script is named run_inference_ring.slurm based on previous steps
  SLURM_SCRIPT_PATH="${CURRENT_REPO_DIR}/testing/run_inference_ring.slurm"
else
  CURRENT_REPO_DIR="$LOCAL_REPO_DIR"
fi
PYTHON_SCRIPT_PATH="${CURRENT_REPO_DIR}/scripts/inference.py" # Point to the main inference script

# Construct absolute default paths
if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  INSOMNIA_BASE_DIR="/insomnia001/depts/edu/COMSE6998/sg3790" # Base directory where llama-hf resides
  DEFAULT_MODEL_ABS_PATH="${INSOMNIA_BASE_DIR}/${DEFAULT_MODEL_REL_PATH}"
  DEFAULT_TOKENIZER_ABS_PATH="${INSOMNIA_BASE_DIR}/${DEFAULT_TOKENIZER_REL_PATH}"
else
  # Assuming model/tokenizer are relative to the local repo dir for local runs
  DEFAULT_MODEL_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_MODEL_REL_PATH}"
  DEFAULT_TOKENIZER_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_TOKENIZER_REL_PATH}"
fi

echo "[INFO] Navigating to repository: $CURRENT_REPO_DIR"
cd "$CURRENT_REPO_DIR" || { echo "[ERROR] Failed to cd to repo"; exit 1; }

# Only pull on Insomnia
if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  echo "[INFO] Pulling latest changes from Git..."
  git pull || { echo "[WARN] Git pull failed, continuing with current code."; }
fi

echo "[INFO] Installing/updating package dependencies with 'pip install -e .'..."
# pip install -e . || { echo "[WARN] pip install failed, attempting to continue..."; }

# Navigate back home to store output files there
cd ~ || { echo "[ERROR] Failed to cd to home"; exit 1; }

# Remove old output files from the home directory (where Slurm outputs) and the testing dir (where local outputs)
echo "[INFO] Removing old ring_infer_*.out files from ~ and testing/ ..."
rm -f ring_infer_*.out "${CURRENT_REPO_DIR}/testing/ring_infer_local_cpu_*.out"

# --- Cleanup Function ---
cleanup() {
    echo "" # Newline after Ctrl+C
    echo "[INFO] Cleaning up..."
    if [[ "$RUN_LOCATION" == "insomnia" ]] && [[ -n "$job_id" ]]; then
        echo "[INFO] Attempting to cancel Slurm job $job_id..."
        scancel "$job_id"
    elif [[ "$RUN_LOCATION" == "local" ]] && [[ -n "$pid" ]]; then
        echo "[INFO] Attempting to kill local process $pid..."
        kill "$pid" 2>/dev/null
    fi
    echo "[INFO] Exiting."
    exit 130
}
trap cleanup SIGINT SIGTERM

# --- Prepare Arguments ---
script_args=("$@") # Use all arguments passed to this script

# Add default model/tokenizer if not provided
has_model_path=$(echo "${script_args[@]}" | grep -q -- '--model_path'; echo $?)
has_tokenizer=$(echo "${script_args[@]}" | grep -q -- '--tokenizer'; echo $?)
if [[ $has_model_path -ne 0 ]]; then
    script_args+=(--model_path "$DEFAULT_MODEL_ABS_PATH")
fi
if [[ $has_tokenizer -ne 0 ]]; then
    script_args+=(--tokenizer "$DEFAULT_TOKENIZER_ABS_PATH")
fi

echo "[INFO] Running Ring Attention Inference"
echo "[INFO] With arguments: ${script_args[*]}"

job_id=""
pid=""

if [[ "$RUN_LOCATION" != "insomnia" ]]; then
    # --- Local CPU Execution ---
    # echo "[WARN] This script is primarily intended for Slurm submission on Insomnia." # Optional warning
    echo "[WARN] Attempting local CPU execution, but ensure environment is correct."
    timestamp=$(date +%Y%m%d_%H%M%S)
    # Save log file inside the testing directory within the repo
    output_file="${CURRENT_REPO_DIR}/testing/ring_infer_local_cpu_${timestamp}.out"
    echo "[INFO] Running local process with torchrun (nproc=2) on CPU. Output: $output_file"
    run_command=(
        "torchrun" "--nproc_per_node=2" # Use 2 processes locally for Ring
        "$PYTHON_SCRIPT_PATH"
        "--architecture" "llama" # Explicitly add architecture
        "--variant" "7b"      # Explicitly add variant
        "--device_type" "cpu" # Changed to CPU
        "--default_dtype" "fp16" # Match runner.sh for CPU test
        "--model_source" "hf" # Add model source hint
        "--no_use_cache"      # Disable KV cache like runner.sh
        "--distributed" \
        "--distributed_strategy" "ring" \
        "${script_args[@]}"
    )
    echo "[INFO] Executing: ${run_command[*]} > $output_file 2>&1 &"
    "${run_command[@]}" > "$output_file" 2>&1 &
    pid=$!
    echo "[SUCCESS] Local process started with PID: $pid"
    wait_command="ps -p $pid"
else
    # --- Insomnia/Slurm Execution ---
    echo "[INFO] Submitting Slurm job via $SLURM_SCRIPT_PATH"
    output=$(sbatch "$SLURM_SCRIPT_PATH" "${script_args[@]}" 2>&1) # Pass args to sbatch
    echo "[INFO] sbatch output: $output"
    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
    if [[ -z "$job_id" ]]; then
        echo "[ERROR] sbatch submission failed."
        exit 1
    fi
    echo "[SUCCESS] Job submitted with ID: $job_id"
    output_file="ring_infer_${job_id}.out" # Match Slurm script output pattern
    wait_command="squeue -u $USER"
fi

echo "[INFO] Job/Process started. Monitor status with: $wait_command"
echo "[INFO] Check output file: $output_file"

# --- Wait for Output and Tail ---
echo "[INFO] Waiting for output file: $output_file ..."
max_wait_loops=12 # Wait up to 1 minute (12 loops * 5 seconds)

for i in $(seq 1 $max_wait_loops); do
  # Check if the file exists and has content (useful for Slurm start delay)
  if [[ -s "$output_file" ]]; then
    echo "" # Newline after dots
    echo "[INFO] Found output file: $output_file"
    echo "[INFO] Tailing output file. Press Ctrl+C to stop tailing (will attempt cleanup)."
    # Use subshell to handle tail interruption gracefully with trap
    ( tail -n +1 -f "$output_file" ) & # Start tailing from the beginning (-n +1)
    tail_pid=$!
    wait $tail_pid # Wait for tail to be interrupted or finish
    exit 0 # Exit successfully after tailing finishes or is interrupted
  fi
  echo -n "." # Print a dot to show we are waiting
  sleep 5
done

echo "" # Newline after dots
echo "[ERROR] Output file $output_file not found or empty after waiting. Check job/process status with: $wait_command"
cleanup # Attempt cleanup on timeout
exit 1
