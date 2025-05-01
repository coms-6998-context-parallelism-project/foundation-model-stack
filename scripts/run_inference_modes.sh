#!/bin/bash

# Script to run FMS inference in different modes.
# Usage: ./scripts/run_inference_modes.sh [standard|ring]

# --- Common Settings ---
# Adjust these variables to change the model, tokenizer, etc.
MODEL_PATH="../llama-hf"  # Adjust path relative to foundation-model-stack root if needed
TOKENIZER_PATH="../llama-hf" # Adjust path relative to foundation-model-stack root if needed
ARCHITECTURE="llama"
VARIANT="7b" # e.g., 7b, 13b, micro
NPROC_PER_NODE_RING=2 # Number of processes for the ring attention test

# --- Mode Selection from Argument ---
MODE=$1

# --- Execution Logic ---
if [ "$MODE" == "standard" ]; then
  # --- Standard Inference (Single Process, MPS) ---
  echo "Running Standard Inference (MPS)..."
  PYTORCH_ENABLE_MPS_FALLBACK=1 torchrun --nproc_per_node=1 \
    scripts/inference.py \
    --architecture ${ARCHITECTURE} \
    --variant ${VARIANT} \
    --model_path ${MODEL_PATH} \
    --model_source hf \
    --tokenizer ${TOKENIZER_PATH} \
    --device_type mps \
    --default_dtype fp16 \
    --no_use_cache

elif [ "$MODE" == "ring" ]; then
  # --- Ring Attention Inference (N Processes, CPU - for testing setup) ---
  echo "Running Ring Attention Inference (CPU, ${NPROC_PER_NODE_RING} procs)..."
  torchrun --nproc_per_node=${NPROC_PER_NODE_RING} \
    scripts/inference.py \
    --architecture ${ARCHITECTURE} \
    --variant ${VARIANT} \
    --model_path ${MODEL_PATH} \
    --model_source hf \
    --tokenizer ${TOKENIZER_PATH} \
    --device_type cpu \
    --default_dtype fp32 \
    --distributed_strategy ring \
    --attn_algorithm ring \
    --no_use_cache

else
  echo "Usage: $0 [standard|ring]"
  echo "Error: Invalid or missing mode specified."
  exit 1
fi