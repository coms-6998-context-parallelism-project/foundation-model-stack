#!/bin/bash
# Script to submit the Ring Attention inference job via SLURM

REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
SLURM_SCRIPT="${REPO_DIR}/testing/run_inference.slurm"

echo "[INFO] Submitting Ring Attention inference job..."
cd "$REPO_DIR" || { echo "[ERROR] Repo directory not found."; exit 1; }

output=$(sbatch "$SLURM_SCRIPT")
echo "[INFO] sbatch output: $output"

job_id=$(echo "$output" | grep -oP 'Submitted batch job \K[0-9]+')
if [[ -n "$job_id" ]]; then
  echo "[SUCCESS] Job submitted with ID: $job_id"
  echo "[INFO] Output will be written to: inference_${job_id}.out"
else
  echo "[ERROR] Job submission failed."
  exit 1
fi
