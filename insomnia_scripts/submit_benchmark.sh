#!/bin/bash
# Benchmark comparison submit script:
# Submits a single Slurm job on Insomnia (GPU) or runs locally (CPU)
# to compare two attention types using intermediate output checks.

# --- Environment Detection ---
RUN_LOCATION="local" # Default to local
if [[ "$(hostname)" == insomnia* ]]; then
  RUN_LOCATION="insomnia"
  echo "[INFO] Detected Insomnia environment. Will use Slurm for GPU execution."
else
  echo "[INFO] Detected Local environment. Will use Python directly for CPU execution."
fi

# --- Base Paths ---
INSOMNIA_REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
LOCAL_REPO_DIR="/Users/sadigulcelik/Documents/CompSci/HPML-2025-Spring/FMSwrapper/foundation-model-stack" # Adjust if your local path differs

# --- Comparison Configuration ---
BASELINE_ATTN_TYPE="sdpa" # The reference implementation
TEST_ATTN_TYPE="ring"     # The implementation to test against the baseline

# --- Set Current Paths ---
if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  CURRENT_REPO_DIR="$INSOMNIA_REPO_DIR"
  SLURM_SCRIPT_PATH="${CURRENT_REPO_DIR}/insomnia_scripts/run_benchmark.slurm"
else
  CURRENT_REPO_DIR="$LOCAL_REPO_DIR"
  PYTHON_SCRIPT_PATH="${CURRENT_REPO_DIR}/scripts/benchmark_attn.py"
fi

echo "[INFO] Navigating to repository: $CURRENT_REPO_DIR"
cd "$CURRENT_REPO_DIR" || { echo "[ERROR] Failed to cd to repo"; exit 1; }

echo "[INFO] Pulling latest changes from Git..."
git pull || { echo "[WARN] Git pull failed, continuing with current code."; }

# Navigate back home to store output files there
cd ~ || { echo "[ERROR] Failed to cd to home"; exit 1; }

echo "[INFO] Removing old benchmark_*.out files..."
if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  rm -f benchmark_compare_*.out benchmark_*.out # Clean up previous Slurm formats
else
  rm -f benchmark_local_cpu_*.out # Clean up previous local formats
fi

if [[ "$RUN_LOCATION" == "insomnia" ]]; then
  # --- Insomnia/Slurm Execution ---
  echo "[INFO] Submitting Slurm benchmark job to compare $BASELINE_ATTN_TYPE (baseline) vs $TEST_ATTN_TYPE"
  job_id=""
  for i in {1..5}; do
    # Pass the specific comparison argument and any other user-provided arguments ($@)
    # The Slurm script will pass these directly to the python script.
    output=$(sbatch "$SLURM_SCRIPT_PATH" \
               --attn_types_to_compare "$BASELINE_ATTN_TYPE" "$TEST_ATTN_TYPE" \
               "$@" 2>&1) # Pass along other args like --max_new_tokens
    echo "[INFO] sbatch output: $output"
    job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
    if [[ -n "$job_id" ]]; then
      echo "[SUCCESS] Job submitted with ID: $job_id"
      break
    fi
    echo "[WARN] sbatch submission failed, retrying in 3s..."
    sleep 3
  done

  if [[ -z "$job_id" ]]; then
    echo "[ERROR] sbatch failed after retries."
    exit 1
  fi
  output_file="benchmark_compare_${job_id}.out" # Match the Slurm script's output pattern
  wait_command="squeue -u $USER" # Command to check job status

else
  # --- Local CPU Execution ---
  echo "[INFO] Running local CPU benchmark job to compare $BASELINE_ATTN_TYPE (baseline) vs $TEST_ATTN_TYPE"
  echo "[WARN] Ring attention comparison on CPU is experimental and may fail."
  # Define output file name (using timestamp for uniqueness)
  timestamp=$(date +%Y%m%d_%H%M%S)
  output_file="benchmark_local_cpu_${timestamp}.out"
  echo "[INFO] Output will be saved to: $output_file"

  # Construct the python command
  # NOTE: Ring attention might fail on CPU or without distributed setup.
  # The python script needs to handle this gracefully or error out.
  python_command=(
      "python"
      "$PYTHON_SCRIPT_PATH"
      "--attn_types_to_compare" "$BASELINE_ATTN_TYPE" "$TEST_ATTN_TYPE"
      "--device_type" "cpu" # Explicitly set CPU
      # Add other necessary defaults if not handled by argparse in python script
      # e.g., --dtype fp32 might be needed for CPU
      "--dtype" "fp32"
  )
  # Add user-provided arguments
  python_command+=("$@")

  echo "[INFO] Executing: ${python_command[*]} > $output_file 2>&1 &"
  # Execute in background, redirect stdout/stderr
  # Ensure the python script has executable permissions if needed, though `python script.py` is usually fine.
  "${python_command[@]}" > "$output_file" 2>&1 &
  pid=$!
  echo "[SUCCESS] Local process started with PID: $pid"
  wait_command="ps -p $pid" # Command to check process status
fi

# --- Wait for Output ---
echo "[INFO] Waiting for output file... $output_file"
max_wait_loops=60 # 60 loops * 5 seconds = 300 seconds (5 minutes)

for i in $(seq 1 $max_wait_loops); do
  if [[ -f "$output_file" ]]; then
    echo "[INFO] Found output file: $output_file"
    # Check if the process/job is still running before tailing indefinitely
    # For local, check PID; for Slurm, check job ID (optional, tail -f might be enough)
    echo "[INFO] Tailing output file. Check for 'Comparison Successful' or 'MISMATCH DETECTED'."
    tail -f "$output_file"
    # Wait for the background process to finish if running locally, after tail is interrupted (Ctrl+C)
    if [[ "$RUN_LOCATION" == "local" ]]; then
        wait $pid
        echo "[INFO] Local process $pid finished with exit code $?."
    fi
    exit 0 # Exit successfully after finding and tailing the file
  fi
  echo -n "." # Progress indicator
  sleep 5
done

# If the loop finishes without finding the file
echo "[ERROR] Output file $output_file not found after waiting. Check job/process status with: $wait_command"
exit 1

exit 0
