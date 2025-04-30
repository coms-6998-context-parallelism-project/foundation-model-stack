#!/bin/bash

# 1. Go to repo
cd /insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack || { echo "Failed to cd to repo"; exit 1; }

# 2. Pull latest changes
echo "Pulling latest changes from git..."
git pull

# 3. Go back to home
cd ~ || { echo "Failed to cd to home"; exit 1; }

# 4. Remove old .out files
echo "Removing old *.out files..."
rm -f *.out

# 5. Submit job
echo "Submitting Slurm inference job..."
scripts/submit_inference.sh

# 6. List files to find new inference output
sleep 2  # Small wait to let slurm output appear
echo "Listing files:"
ls -t | head -10

# 7. Auto-detect latest inference output and tail it
latest_out=$(ls -t inference_*.out 2>/dev/null | head -1)
if [ -z "$latest_out" ]; then
  echo "No inference_*.out file found!"
else
  echo "Tailing latest output: $latest_out"
  tail -f "$latest_out"
fi