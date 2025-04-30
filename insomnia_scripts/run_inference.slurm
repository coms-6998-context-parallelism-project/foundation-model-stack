#!/bin/bash
#
#SBATCH --account=edu
#SBATCH --job-name=llama_inference_cpu
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --time=00:3:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:2
#SBATCH --output=inference_%j.out
echo "Starting inference job at $(date)"

echo "=== Job started at $(date) on $(hostname) ==="
which python
echo "CPU cores: $(nproc)"
free -h
top -b -n 1 | head -20

# Begin background system monitoring every 60 seconds
(
  while true; do
    echo "[MONITOR] $(date)" >> resource_log.txt
    nvidia-smi >> resource_log.txt
    top -b -n 1 | head -20 >> resource_log.txt
    echo "-----------------------------" >> resource_log.txt
    sleep 60
  done
) &

# 3. Navigate to repo
cd /insomnia001/depts/edu/COMSE6998/sg3790/foundation-model-stack
pip install --user -e . --quiet
pip install --user safetensors --quiet
pip install --user transformers --quiet


# 5. Run inference (CPU mode)
torchrun --nproc_per_node=2 \
  scripts/inference.py \
  --architecture llama \
  --variant 7b \
  --model_path /insomnia001/depts/edu/COMSE6998/sg3790/llama-hf \
  --model_source hf \
  --tokenizer /insomnia001/depts/edu/COMSE6998/sg3790/llama-hf \
  --device_type cuda \
  --default_dtype fp16 \
  --no_use_cache --distributed \
  --distributed_strategy ring \
  --attn_algorithm ring

echo "Inference job completed at $(date)"