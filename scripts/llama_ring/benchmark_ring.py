import argparse
import os
import statistics
import random
import warnings
import time
import gc # Make sure gc is imported
import torch
import numpy as np
import torch.distributed as dist
import psutil # For CPU memory
from pathlib import Path

from fms import models
from fms.utils import tokenizers
from fms.distributed.strategy import NoOpStrategy

def print0(*args, **kwargs):
    if int(os.getenv("RANK", 0)) == 0:
        print(*args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark token generation with LLaMA attention strategies")

    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[2]
    model_dir = repo_dir.parent / "llama-hf"
    tokenizer_path = model_dir / "tokenizer.model"

    parser.add_argument("--device_type", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--architecture", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="7b")
    parser.add_argument("--model_path", type=str, default=str(model_dir))
    parser.add_argument("--tokenizer", type=str, default=str(tokenizer_path))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_tokens_to_benchmark", type=int, default=5)
    parser.add_argument("--run_ring_first", action="store_true")
    parser.add_argument("--no-run_ring_first", dest="run_ring_first", action="store_false")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.set_defaults(run_ring_first=True)
    return parser.parse_args()

def set_determinism():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def setup_model(args, strategy=None, dtype=None):
    dist_strategy_param = strategy
    if strategy is NoOpStrategy:
        dist_strategy_param = NoOpStrategy
    return models.get_model(
        args.architecture,
        args.variant,
        model_path=args.model_path,
        device_type=args.device_type,
        source="hf",
        distributed_strategy=dist_strategy_param,
        data_type=dtype
    )

def get_memory_snapshot(device_type, rank=0):
    """Gets a snapshot of current and peak memory usage."""
    if rank != 0: # Memory reporting primarily for rank 0
        return {}

    process = psutil.Process(os.getpid())
    rss_mb = process.memory_info().rss / (1024 * 1024)
    snapshot = {"cpu_rss_mb": rss_mb}

    if device_type == "cuda":
        snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        snapshot["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        snapshot["cuda_max_reserved_mb"] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    elif device_type == "mps":
        snapshot["mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024 * 1024)
        # MPS doesn't have a direct equivalent for max_memory_allocated easily accessible here
    return snapshot

def print_memory_snapshot(label, snapshot, rank=0):
    if rank == 0 and snapshot: # Ensure snapshot is not empty (e.g., from non-rank 0)
        print0(f"[Memory ({label})] CPU RSS: {snapshot.get('cpu_rss_mb', 0):.2f} MB")
        if "cuda_allocated_mb" in snapshot:
            print0(f"  CUDA Allocated: {snapshot['cuda_allocated_mb']:.2f} MB | Max Allocated: {snapshot['cuda_max_allocated_mb']:.2f} MB")
            print0(f"  CUDA Reserved: {snapshot['cuda_reserved_mb']:.2f} MB | Max Reserved: {snapshot['cuda_max_reserved_mb']:.2f} MB")
        elif "mps_allocated_mb" in snapshot:
            print0(f"  MPS Allocated: {snapshot['mps_allocated_mb']:.2f} MB")

def run_generation_benchmark(model, tokenizer, initial_ids, num_tokens_to_gen, label, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    print0(f"\n[Benchmark] {label}: Generating {num_tokens_to_gen} tokens...")
    current_ids = initial_ids.clone()
    past_key_value_states = None
    token_times, generated_text = [], []
    
    # Memory profiling for generation
    if device.type == "cuda" and rank == 0:
        torch.cuda.reset_peak_memory_stats() # Reset peak for this specific generation run
    
    mem_before_gen = get_memory_snapshot(device.type, rank)
    for i in range(num_tokens_to_gen):
        input_ids = current_ids if past_key_value_states is None else current_ids[:, -1:]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        logits = model(input_ids, past_key_value_states=past_key_value_states, use_cache=False)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = (time.perf_counter() - start) * 1000
        token_times.append(duration)

        token_str = tokenizer.convert_ids_to_tokens([next_token.item()])
        token_text = tokenizer.convert_tokens_to_string(token_str)
        generated_text.append(token_text)

        current_ids = torch.cat([current_ids, next_token], dim=1)

    mem_after_gen = get_memory_snapshot(device.type, rank)
    print_memory_snapshot(f"During Generation ({label}) - Before", mem_before_gen, rank)
    print_memory_snapshot(f"During Generation ({label}) - After (Peak for CUDA)", mem_after_gen, rank)
    if rank == 0:
        avg_time = statistics.mean(token_times)
        median_time = statistics.median(token_times)
        print0(f"\nGenerated Sequence ({label}): {'-'.join(generated_text)}")
        print0("Token Timings:")
        for i, t in enumerate(token_times):
            printable_token = generated_text[i].encode('unicode_escape').decode('utf-8')
            print0(f"  * Token {i+1} ({printable_token}): {t:.2f} ms")
        print0(f"\nSummary for {label}:\n  Average: {avg_time:.2f} ms\n  Median: {median_time:.2f} ms\n  Total: {sum(token_times):.2f} ms")

def main():
    args = parse_args()
    set_determinism()

    # --- Enhanced Diagnostic Information ---
    print0("--- Benchmark Arguments ---")
    for arg, value in sorted(vars(args).items()):
        print0(f"  {arg}: {value}")
    print0("---------------------------")

    print0("--- System Configuration ---")
    print0(f"  PyTorch Version: {torch.__version__}")
    if args.device_type == "cuda" and torch.cuda.is_available():
        print0(f"  CUDA Version: {torch.version.cuda}")
        try:
            print0(f"  NCCL Version: {torch.cuda.nccl.version()}")
        except AttributeError:
            print0("  NCCL Version: Not available via torch.cuda.nccl.version()")
    print0("---------------------------")

    print0("--- Relevant Environment Variables ---")
    env_vars_to_log = ["OMP_NUM_THREADS", "NCCL_DEBUG", "NCCL_P2P_DISABLE", "TORCH_DISTRIBUTED_DEBUG", "CUDA_VISIBLE_DEVICES",
                       "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS_PER_NODE", "SLURM_PROCID", "SLURM_LOCALID"]
    for var in env_vars_to_log:
        value = os.getenv(var)
        if value is not None:
            print0(f"  {var}: {value}")
    print0("----------------------------------")
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    if args.device_type == "mps":
        if not torch.backends.mps.is_available():
            raise EnvironmentError("MPS not available.")
        if world_size > 1:
            raise RuntimeError("Distributed MPS not supported.")
        device = torch.device("mps")
    elif world_size > 1:
        if args.device_type == "cuda" and not torch.cuda.is_available():
            raise EnvironmentError("CUDA requested but not available.")
        
        print(f"[Rank {rank}/{world_size}] Initializing distributed process group...")
        if args.device_type == "cuda":
            torch.cuda.set_device(local_rank)
            print(f"[Rank {rank}/{world_size}] Set CUDA device to: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'N/A'})")
            backend = "nccl"
        else:
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
            print(f"[Rank {rank}/{world_size}] Distributed process group initialized with backend '{backend}'.")
        device = torch.device(args.device_type, local_rank)
        print(f"[Rank {rank}/{world_size}] Using device: {device}")

    else:
        device = torch.device(args.device_type)

    try:
        parsed_dtype = getattr(torch, args.dtype)
        # Initial memory snapshot after basic setup and before heavy lifting
        # Placed here to capture state before any major torch operations beyond dtype parsing
        initial_mem_snapshot = get_memory_snapshot(args.device_type, rank)
        print_memory_snapshot("Initial Script Load", initial_mem_snapshot, rank)
    except AttributeError:
        print0(f"[WARNING] Invalid dtype '{args.dtype}', defaulting to float32.")
        parsed_dtype = torch.float32

    torch.set_default_dtype(parsed_dtype)

    tokenizer = tokenizers.get_tokenizer(args.tokenizer) if rank == 0 else None
    if world_size > 1:
        dist.barrier()
        if rank != 0:
            tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    strategies = [("Ring Attention", "ring"), ("Regular Attention", NoOpStrategy)]
    if not args.run_ring_first:
        strategies.reverse()

    prompt_n_values = [10, 20, 50, 100, 200, 300, 400]

    for strategy_label, strategy in strategies:

        if args.device_type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print0(f"\n=== Benchmarking: {strategy_label} ===")
        should_run = strategy is not NoOpStrategy or rank == 0
        model = None
        mem_before_current_model_load = get_memory_snapshot(args.device_type, rank)
        print_memory_snapshot(f"Before Model Load ({strategy_label})", mem_before_current_model_load, rank)

        if should_run:
            warnings.filterwarnings("ignore", message=r"Keys from checkpoint.*rotary_emb\.inv_freq")
            model = setup_model(args, strategy, dtype=parsed_dtype).eval()
            torch.set_grad_enabled(False)
            mem_after_model_load = get_memory_snapshot(args.device_type, rank)
            print_memory_snapshot(f"After Model Load ({strategy_label})", mem_after_model_load, rank)
            if hasattr(model, 'config'):
                print0(f"--- Model Config ({strategy_label}) ---")
                print0(f"  Architecture: {args.architecture}, Variant: {args.variant}")
                print0(f"  Layers: {getattr(model.config, 'nlayers', 'N/A')}, Heads: {getattr(model.config, 'nheads', 'N/A')}, Hidden Dim: {getattr(model.config, 'emb_dim', getattr(model.config, 'dim', 'N/A'))}")
                print0(f"  Vocab Size: {getattr(model.config, 'src_vocab_size', 'N/A')}")
                print0("--------------------------")

            warnings.resetwarnings()

        if world_size > 1:
            dist.barrier()

        for n_val in prompt_n_values:
            prompt_text = ", ".join(map(str, range(n_val)))
            if rank == 0:
                tokens = tokenizer.tokenize(prompt_text)
                prompt_ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).repeat(args.batch_size, 1)
                ids = ids_tensor.to(device)
            else:
                ids = None

            if world_size > 1:
                shape_list = [None]
                if rank == 0:
                    shape_list[0] = ids.shape
                dist.broadcast_object_list(shape_list, src=0)

                if rank != 0:
                    ids = torch.empty(shape_list[0], dtype=torch.long, device=device)
                dist.broadcast(ids, src=0)
                dist.barrier()

            if rank == 0:
                print0(f"\n-- Prompt N={n_val} for {strategy_label} --")
                print0(f"[INFO] Prompt: '{prompt_text}'")
                print0(f"[INFO] Tokens: {ids.shape[1]}, Batch: {args.batch_size}")

            if should_run:
                run_generation_benchmark(model, tokenizer, ids, args.num_tokens_to_benchmark, f"{strategy_label} (N={n_val})", device)

            if world_size > 1:
                dist.barrier()

        if model is not None:
            mem_before_model_del = get_memory_snapshot(args.device_type, rank)
            print_memory_snapshot(f"Before Model Deletion ({strategy_label})", mem_before_model_del, rank)
            del model
            gc.collect() # Explicitly run garbage collection
            if args.device_type == "cuda":
                torch.cuda.empty_cache()
            # Snapshot after deletion and cache clearing
            mem_after_model_del = get_memory_snapshot(args.device_type, rank)
            print_memory_snapshot(f"After Model Deletion ({strategy_label})", mem_after_model_del, rank)
        if world_size > 1:
            dist.barrier()

if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
