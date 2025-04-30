#!/bin/bash
REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
SLURM_SCRIPT_PATH="$HOME/insomnia_scripts/run_inference.slurm"

echo "[INFO] Pulling repo updates..."
cd "$REPO_DIR" && git pull && cd ~

rm -f inference_*.out
echo "[INFO] Submitting inference job..."
job_id=""
for i in {1..5}; do
  output=$(sbatch "$SLURM_SCRIPT_PATH" "$@" 2>&1)
  echo "[INFO] sbatch output: $output"
  job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
  if [[ -n "$job_id" ]]; then
    echo "[SUCCESS] Job submitted with ID: $job_id"
    break
  fi
  echo "[WARN] sbatch retry $i/5..."
  sleep 3
done

if [[ -z "$job_id" ]]; then
  echo "[ERROR] Submission failed after retries"
  exit 1
fi

for i in {1..10}; do
  latest_out=$(ls -t inference_*.out 2>/dev/null | head -1)
  if [[ -n "$latest_out" ]]; then
    echo "[INFO] Monitoring: $latest_out"
    tail -f "$latest_out"
    exit 0
  fi
  echo "[INFO] Waiting for output... ($i)"
  sleep 2
done