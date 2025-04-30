#!/bin/bash
# Enhanced benchmark submit script with sbatch wait and retry logic

REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
SLURM_SCRIPT_PATH="$HOME/slurm/run_benchmark.slurm"

echo "[INFO] Navigating to repository: $REPO_DIR"
cd "$REPO_DIR" || { echo "[ERROR] Failed to cd to repo"; exit 1; }

echo "[INFO] Pulling latest changes..."
git pull

echo "[INFO] Navigating back home..."
cd ~ || { echo "[ERROR] Failed to cd to home"; exit 1; }

echo "[INFO] Removing old benchmark_*.out files..."
rm -f benchmark_*.out

echo "[INFO] Submitting Slurm benchmark job..."
job_id=""
for i in {1..5}; do
  output=$(sbatch "$SLURM_SCRIPT_PATH" "$@" 2>&1)
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

echo "[INFO] Waiting for job output..."
for i in {1..10}; do
  latest_out=$(ls -t benchmark_*.out 2>/dev/null | head -1)
  if [[ -n "$latest_out" ]]; then
    echo "[INFO] Found output file: $latest_out"
    tail -f "$latest_out"
    exit 0
  fi
  echo "[INFO] Waiting for Slurm to write output... ($i)"
  sleep 2
done

echo "[ERROR] Output file not found. Check with: squeue -u $USER"