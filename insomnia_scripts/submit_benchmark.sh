#!/bin/bash
# Benchmark comparison submit script: Submits a single job to compare two attention types.

REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
SLURM_SCRIPT_PATH="${REPO_DIR}/insomnia_scripts/run_benchmark.slurm"

# --- Comparison Configuration ---
BASELINE_ATTN_TYPE="sdpa" # The reference implementation
TEST_ATTN_TYPE="ring"     # The implementation to test against the baseline

echo "[INFO] Navigating to repository: $REPO_DIR"
cd "$REPO_DIR" || { echo "[ERROR] Failed to cd to repo"; exit 1; }

echo "[INFO] Pulling latest changes..."
# Force pull by fetching and resetting hard to the remote branch (e.g., origin/main)
git fetch origin
git reset --hard origin/ring_from_scratch # Adjust 'main' if your main branch has a different name (e.g., master)

echo "[INFO] Navigating back home..."
cd ~ || { echo "[ERROR] Failed to cd to home"; exit 1; }

echo "[INFO] Removing old benchmark_*.out files..."
rm -f benchmark_compare_*.out benchmark_*.out # Clean up previous formats

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

echo "[INFO] Waiting for job output... Submitted Job ID: $job_id"
max_wait_loops=60 # 60 loops * 5 seconds = 300 seconds (5 minutes)
output_file="benchmark_compare_${job_id}.out" # Match the Slurm script's output pattern

for i in $(seq 1 $max_wait_loops); do
  if [[ -f "$output_file" ]]; then
    echo "[INFO] Found output file: $output_file"
    echo "[INFO] Tailing output file. Check for 'Comparison Successful' or 'MISMATCH DETECTED'."
    tail -f "$output_file"
    exit 0 # Exit successfully after finding and tailing the file
  fi
    if [[ $i -eq $max_wait_loops ]]; then
       echo "[WARN] Timeout waiting for $expected_file"
    else
       echo -n "." # Progress indicator
       sleep 5
    fi
  fi

# If the loop finishes without finding the file
echo "[ERROR] Output file $output_file not found after waiting. Check job status with: squeue -u $USER"
exit 1

exit 0