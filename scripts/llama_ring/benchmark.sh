#!/bin/bash

# Define the sequence lengths to iterate over
seq_lengths=(10 20 50 100 200 300 400 500 1000 2000 4000)

# Loop through each sequence length and submit the job
for seq_len in "${seq_lengths[@]}"; do
    output_file="benchmarks_1/SEQ_${seq_len}.txt"
    echo "Submitting job for sequence length ${seq_len}..."
    sbatch --output="$output_file" foundation-model-stack/scripts/llama_ring/benchmark_ring.slurm "$seq_len"
done