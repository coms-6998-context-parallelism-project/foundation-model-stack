#!/bin/bash
# Comparison submit script:
# Submits a single Slurm job (Insomnia/GPU) or runs a local process (CPU/MPS)
# to execute run_comparison.py, which compares two attention algorithms internally.

# --- Environment Detection ---
INSOMNIA_REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
if [[ -d "$INSOMNIA_REPO_DIR" ]]; then
  RUN_LOCATION="insomnia"
  echo "[INFO] Detected Insomnia environment. Will use Slurm for GPU execution."
else
  RUN_LOCATION="local"
  echo "[INFO] Detected Local environment. Will use Python directly for CPU/MPS execution."
fi

# --- Base Paths ---
LOCAL_REPO_DIR="/Users/sadigulcelik/Documents/CompSci/HPML-2025-Spring/FMSwrapper/foundation-model-stack" # Adjust if needed

# --- Default Comparison Configuration ---
BASELINE_ALGO="sdpa"
TEST_ALGO="ring"

# --- Default Paths (relative to repo root) ---
DEFAULT_MODEL_REL_PATH="llama-hf"
DEFAULT_TOKENIZER_REL_PATH="llama-hf/tokenizer.model"

# --- Set Current Paths ---
if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  CURRENT_REPO_DIR="$INSOMNIA_REPO_DIR"
  SLURM_SCRIPT_PATH="${CURRENT_REPO_DIR}/testing/run_comparison.slurm" # Path to the Slurm script
else
  CURRENT_REPO_DIR="$LOCAL_REPO_DIR"
fi
PYTHON_SCRIPT_PATH="${CURRENT_REPO_DIR}/testing/run_comparison.py"

# Construct absolute default paths
DEFAULT_MODEL_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_MODEL_REL_PATH}"
DEFAULT_TOKENIZER_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_TOKENIZER_REL_PATH}"

echo "[INFO] Navigating to repository: $CURRENT_REPO_DIR"
cd "$CURRENT_REPO_DIR" || { echo "[ERROR] Failed to cd to repo"; exit 1; }

echo "[INFO] Pulling latest changes from Git..."
git pull || { echo "[WARN] Git pull failed, continuing with current code."; }

echo "[INFO] Installing/updating package dependencies with 'pip install -e .'..."
pip install -e . || { echo "[WARN] pip install failed, attempting to continue..."; }

# Navigate back home to store output files there
cd ~ || { echo "[ERROR] Failed to cd to home"; exit 1; }

echo "[INFO] Removing old comparison_*.out files..."
rm -f comparison_*.out

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
script_args=()
user_provided_compare=false
# Separate comparison args from others
other_args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --compare)
      BASELINE_ALGO="$2"
      TEST_ALGO="$3"
      user_provided_compare=true
      shift 3 # Consume --compare, baseline, test
      ;;
    *)
      other_args+=("$1") # Store other args
      shift
      ;;
  esac
done

# Add default model/tokenizer if not in other_args
has_model_path=$(echo "${other_args[@]}" | grep -q -- '--model_path'; echo $?)
has_tokenizer=$(echo "${other_args[@]}" | grep -q -- '--tokenizer'; echo $?)
if [[ $has_model_path -ne 0 ]]; then
    other_args+=(--model_path "$DEFAULT_MODEL_ABS_PATH")
fi
if [[ $has_tokenizer -ne 0 ]]; then
    other_args+=(--tokenizer "$DEFAULT_TOKENIZER_ABS_PATH")
fi

# Final script arguments
script_args=(--compare "$BASELINE_ALGO" "$TEST_ALGO" "${other_args[@]}")

echo "[INFO] Running comparison: Baseline='${BASELINE_ALGO}', Test='${TEST_ALGO}'"
echo "[INFO] With arguments: ${other_args[*]}"

job_id=""
pid=""

if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  # --- Insomnia/Slurm Execution ---
  echo "[INFO] Submitting Slurm job via $SLURM_SCRIPT_PATH"
  # Pass the collected arguments directly to the Slurm script
  sbatch_args=("${script_args[@]}")

  # Use sbatch to submit the dedicated Slurm script
  output=$(sbatch "$SLURM_SCRIPT_PATH" "${sbatch_args[@]}" 2>&1)
  echo "[INFO] sbatch output: $output"
  job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
  if [[ -z "$job_id" ]]; then
      echo "[ERROR] sbatch submission failed."
      exit 1
  fi
  echo "[SUCCESS] Slurm job submitted with ID: $job_id"
  # Construct the expected output filename based on the Slurm script's --output pattern
  output_file="comparison_${job_id}.out"
  wait_command="squeue -u $USER"

else
  # --- Local CPU/MPS Execution ---
  timestamp=$(date +%Y%m%d_%H%M%S)
  output_file="comparison_local_${BASELINE_ALGO}_vs_${TEST_ALGO}_${timestamp}.out"
  echo "[INFO] Running local process. Output: $output_file"
  # Determine device for local run
  local_device="cpu"
  # Simple check for Metal support on macOS
  if [[ "$(uname)" == "Darwin" ]] && system_profiler SPDisplaysDataType | grep -q "Metal"; then
      local_device="mps"
      echo "[INFO] Metal GPU detected, using device_type=mps"
  fi
  # Use torchrun for consistency, even locally (nproc=1 unless ring on GPU)
  run_command=(
      "torchrun" "--nproc_per_node=1" # Always 1 for local CPU/MPS
      "$PYTHON_SCRIPT_PATH"
      "--device_type" "$local_device"
      "--dtype" "fp32" # fp32 safer for CPU/MPS
      "${script_args[@]}"
  )
  echo "[INFO] Executing: ${run_command[*]} > $output_file 2>&1 &"
  "${run_command[@]}" > "$output_file" 2>&1 &
  pid=$!
  echo "[SUCCESS] Local process started with PID: $pid"
  wait_command="ps -p $pid"
fi

# --- Wait for Output ---
echo "[INFO] Waiting for output file: $output_file ..."
max_wait_loops=120 # Increased wait time (loops * 5 seconds)

for i in $(seq 1 $max_wait_loops); do
  if [[ -f "$output_file" ]]; then
    echo ""
    echo "[INFO] Found output file: $output_file"
    echo "[INFO] Tailing output file. Check for comparison results."
    echo "Press Ctrl+C to stop tailing (will attempt cleanup)."
    # Use subshell to handle tail interruption gracefully with trap
    ( tail -f "$output_file" ) &
    tail_pid=$!
    wait $tail_pid # Wait for tail to be interrupted or finish
    exit 0
  fi
  echo -n "."
  sleep 5
done

echo "" # Newline after dots
echo "[ERROR] Output file $output_file not found after waiting. Check job/process status with: $wait_command"
cleanup # Attempt cleanup on timeout
exit 1

```