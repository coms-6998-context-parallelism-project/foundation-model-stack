# HPML Project: Implementing Context Parallelism in IBM's FMS using Ring Attention

## Team Information

* **Team Name**: FMS-Ring
* **Members**:

  * Sadi Gulcelik (sg3790)
  * Joshua Mathew (jm5915)
  * Jaewon Lee (jl6367)

---

## 1. Problem Statement

Transformer-based large language models (LLMs) struggle with inference performance and memory use at long context lengths, due to the quadratic scaling of self-attention. We implement **Ring Attention**, a context-parallel self-attention strategy, within **IBM’s Foundation Model Stack (FMS)** to enable scalable inference across multiple GPUs. The goal is to alleviate memory bottlenecks and reduce latency for long sequences without modifying model weights or accuracy.

---

## 2. Model Description

We use the **LLaMA 3 8B** model within IBM’s Foundation Model Stack (FMS). Key elements include:

* **Framework**: PyTorch (via IBM FMS)
* **Base Model**: LLaMA 3 8B (\~16 GB in FP32)
* **Main Components**:
  * `**RingAttentionStrategy**`: Orchestrates key/value sharding and inter-GPU communication in a ring topology
  * `**LlamaRing**`: Implements attention computation across the ring
* **Test Environments**:
  * 2× NVIDIA L40S (Columbia Insomnia cluster)
  * 4× NVIDIA L40S (external NVLink cluster)

---

## 3. Final Results Summary

| Metric                    | Value                                           |
| ------------------------- | ----------------------------------------------- |
| Final Accuracy            | Matches single-GPU baseline outputs             |
| Max Context Length Tested | 6998 tokens                                     |
| Inference Latency (FP32)  | \~6000 ms (baseline) → \~4000 ms (Ring, 4 GPUs) |
| Model Size                | \~16 GB (FP32)                                  |
| Training Time/Epoch       | N/A (inference-only project)                    |
| Device                    | 2× / 4× NVIDIA L40S                             |

---

## 4. Reproducibility Instructions

### A. Requirements

Install all dependencies from the root of the FMS repository:

```bash
pip install -e .
```

---

### B. Wandb Dashboard

Not used. Logs and performance metrics are written to `.out` files and `.csv` logs for analysis.

---

### C. Inference Only

This is an inference-only project. Use the following commands:

This is an inference-only project. To run inference from the root of the repository, use the following command:

```bash
torchrun --nproc_per_node=1 foundation-model-stack/scripts/inference.py --architecture llama --variant 7b --model_path ../llama-hf --model_source hf --tokenizer ../llama-hf/tokenizer.model --device_type mps --default_dtype fp16 --no_use_cache
```

**Notes:**
* This command should be run from the root of the FMS repository.
* `--model_path ../llama-hf` and `--tokenizer ../llama-hf/tokenizer.model` are relative paths. Ensure these point correctly to your model and tokenizer files from the repository root.
* Key arguments:
    * `--nproc_per_node`: Number of processes (e.g., GPUs) per node.
    * `--default_dtype`: Data type for model weights and computations (e.g., `fp16`, `float32`).
    * `--device_type`: The type of device to run on (e.g., `mps` for Apple Silicon, `cuda` for NVIDIA GPUs).

---

### D. Evaluation

Benchmarking runs generate structured `.csv` output files using the `**sg_fast_benchmarking**` branch of our repo.
We parse these in a **Jupyter notebook** for all latency and scaling visualizations.

---

### E. Quickstart: Minimum Reproducible Result

```bash
# Step 0: Clone the Repo

# Step 1: Install FMS package
pip install -e 

# Step 2: Enter benchmark directory
cd /scripts/llama_ring



# Step 3: Run the benchmark
./benchmark.sh --nproc 2 ring
```

**Logs:**

* On Slurm: `~/inference_insomnia_<job_id>.out`
* Locally: `testing/inference_local_<timestamp>.out`

---

## 5. Notes

* `RingAttentionStrategy` and `LlamaRing` are our primary contributions to the FMS attention stack.
* The code builds on IBM FMS’s support for FlashAttention and distributed inference.
* Benchmark analysis is performed using a Jupyter notebook on `.csv` logs.
* For cluster-specific jobs and extensions, see the `sg_fast_benchmarking` Git branch.
