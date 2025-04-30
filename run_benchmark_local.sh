#!/bin/bash

# Script to run the attention benchmark locally on Mac

echo "====================================="
echo "Running Benchmark Locally..."
echo "====================================="

# Ensure we are in the script's directory (optional, but good practice)
# cd "$(dirname "$0")" || exit 1

# Define local paths and settings
LOCAL_MODEL_PATH="./llama-hf" # Adjust if your local path is different
LOCAL_TOKENIZER_PATH="./llama-hf" # Adjust if your local path is different
LOCAL_DEVICE="cpu" # Or "mps" if you want to try that

# Set environment variables if needed
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Run the Python benchmark script
python scripts/benchmark_attn.py \
    --model_path "${LOCAL_MODEL_PATH}" \
    --tokenizer "${LOCAL_TOKENIZER_PATH}" \
    --device_type "${LOCAL_DEVICE}" \
    --dtype fp16 \
    --max_new_tokens 3 10 # Example: test a couple of token lengths

echo "====================================="
echo "Local benchmark run complete."
echo "====================================="