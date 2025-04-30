#!/bin/bash
# This script runs both the Ring Attention and Regular Attention inference commands.

echo "====================================="
echo "Running Ring Attention (Distributed)..."
echo "====================================="
PYTORCH_ENABLE_MPS_FALLBACK=1 torchrun --nproc_per_node=2 \
  scripts/inference.py \
  --architecture llama \
  --variant 7b \
  --model_path ./llama-hf \
  --model_source hf \
  --tokenizer ./llama-hf \
  --device_type cpu \
  --default_dtype fp16 \
  --no_use_cache \
  --distributed \
  --distributed_strategy ring \
  --attn_algorithm ring

echo ""
echo "====================================="
echo "Running Regular Attention (Non-Distributed)..."
echo "====================================="
PYTORCH_ENABLE_MPS_FALLBACK=1 torchrun --nproc_per_node=1 \
  scripts/inference.py \
  --architecture llama \
  --variant 7b \
  --model_path ./llama-hf \
  --model_source hf \
  --tokenizer ./llama-hf \
  --device_type cpu \
  --default_dtype fp16 \
  --no_use_cache

echo ""
echo "====================================="
echo "Both runs complete."
echo "====================================="