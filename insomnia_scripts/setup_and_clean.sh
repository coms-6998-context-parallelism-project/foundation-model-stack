#!/bin/bash

# === 1. Create a clean directory structure ===
echo "Creating project structure..."

mkdir -p ~/project/{scripts,logs,archive,slurm}
mv run_*.slurm slurm/
mv *.sh scripts/
mv *.out logs/ 2>/dev/null || echo "No .out files to move"
mv resource_log.txt logs/ 2>/dev/null || echo "No resource_log.txt found"

# Backup old files
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p archive/$timestamp
cp slurm/*.slurm scripts/*.sh logs/*.out logs/resource_log.txt archive/$timestamp 2>/dev/null

echo "Files backed up to archive/$timestamp"

# === 2. Update submit_benchmark.sh to handle slow sbatch ===
cat > scripts/submit_benchmark.sh <<'EOF'
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
EOF
chmod +x scripts/submit_benchmark.sh

# === 3. Optionally enhance run_all.sh the same way ===
sed -i 's/sbatch run_inference.slurm/scripts\/submit_inference.sh/' scripts/run_all.sh

# === 4. Create a cleaner submit_inference.sh (optional symmetry) ===
cat > scripts/submit_inference.sh <<'EOF'
#!/bin/bash
REPO_DIR="/insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack"
SLURM_SCRIPT_PATH="$HOME/slurm/run_inference.slurm"

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
EOF
chmod +x scripts/submit_inference.sh

echo "[DONE] Setup complete. Use 'scripts/submit_benchmark.sh' or 'scripts/submit_inference.sh'"