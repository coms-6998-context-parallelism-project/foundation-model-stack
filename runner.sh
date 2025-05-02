#!/bin/bash

# Script to run ring attention (nproc=2) and/or single-process attention (nproc=1)

# Default behavior: run both
run_ring=true
run_single=true

# If any arguments are provided, assume user wants specific runs
if [ "$#" -gt 0 ]; then
  run_ring=false
  run_single=false
fi

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ring) run_ring=true ;;
        --single) run_single=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run Ring Attention if selected
if [ "$run_ring" = true ]; then
  echo "====================================="
  echo "Running Ring Attention (nproc=3)..."
  echo "====================================="
  PYTORCH_ENABLE_MPS_FALLBACK=1 torchrun --nproc_per_node=3 \
    scripts/inference.py \
    --architecture llama \
    --variant 7b \
    --model_path ../llama-hf \
    --model_source hf \
    --tokenizer ../llama-hf \
    --device_type cpu \
    --default_dtype fp16 \
    --distributed_strategy ring \
    --no_use_cache --distributed
  echo ""
fi

# Run Single Process if selected
if [ "$run_single" = true ]; then
  echo "====================================="
  echo "Running Single Process (nproc=1)..."
  echo "====================================="
  torchrun --nproc_per_node=1 \
    scripts/inference.py \
    --architecture llama \
    --variant 7b \
    --model_path ../llama-hf \
    --model_source hf \
    --tokenizer ../llama-hf \
    --device_type mps \
    --default_dtype fp16 \
    --no_use_cache
  echo ""
fi

echo ""
echo "Done."