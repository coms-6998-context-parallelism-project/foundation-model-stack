#!/usr/bin/env bash
# Run benchmark locally using torchrun and multiple GPUs

set -eo pipefail
IFS=$'\n\t'

# --- Base Paths ---
CURRENT_REPO_DIR="$(pwd)"
DEFAULT_MODEL_REL_PATH="../llama-hf"
DEFAULT_TOKENIZER_REL_PATH="../llama-hf/tokenizer.model"
DEFAULT_MODEL_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_MODEL_REL_PATH}"
DEFAULT_TOKENIZER_ABS_PATH="${CURRENT_REPO_DIR}/${DEFAULT_TOKENIZER_REL_PATH}"

# --- Detect CUDA ---
if command -v nvidia-smi &>/dev/null && python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "[INFO] CUDA detected — running on GPU."
  DEVICE_TYPE="cuda"
else
  echo "[INFO] CUDA not available — falling back to CPU."
  DEVICE_TYPE="cpu"
fi

# --- Install local package if needed ---
echo "[INFO] pip install -e ."
pip install -e . >/dev/null 2>&1 || echo "[WARN] pip install failed"

# --- Prepare log dir ---
mkdir -p "${CURRENT_REPO_DIR}/testing"
cd "$CURRENT_REPO_DIR"

# --- Cleanup old logs ---
echo "[INFO] Cleaning old outputs..."
rm -f testing/inference_local_*.out

# --- Parse script arguments ---
passthrough_args=()
nproc_value=2 # default

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc)
      nproc_value="$2"
      shift 2
      ;;
    *)
      passthrough_args+=("$1")
      shift
      ;;
  esac
done

# Add defaults if not provided
if ! printf '%s\n' "${passthrough_args[@]}" | grep -q -- '--model_path'; then
  passthrough_args+=(--model_path "$DEFAULT_MODEL_ABS_PATH")
fi
if ! printf '%s\n' "${passthrough_args[@]}" | grep -q -- '--tokenizer'; then
  passthrough_args+=(--tokenizer "$DEFAULT_TOKENIZER_ABS_PATH")
fi

# --- Run benchmark ---
timestamp=$(date +%Y%m%d_%H%M%S)
output_file="${CURRENT_REPO_DIR}/testing/inference_local_${timestamp}.out"
echo "[INFO] torchrun (nproc=$nproc_value) → $output_file"

torchrun --nproc_per_node="$nproc_value" \
  scripts/llama_ring_sg/benchmark_ring.py \
  --architecture llama \
  --variant 7b \
  --device_type "$DEVICE_TYPE" \
  --dtype float16 \
  "${passthrough_args[@]}" \
  >"$output_file" 2>&1 &

pid=$!
echo "[SUCCESS] torchrun PID=$pid"
echo "[INFO] Monitor: ps -p $pid"
echo "[INFO] Log file: $output_file"

# Wait for file to appear and start streaming it
for i in {1..50}; do
  [[ -s "$output_file" ]] && break
  sleep 2
done

echo "[INFO] --- Output Start ---"
tail -f "$output_file"
