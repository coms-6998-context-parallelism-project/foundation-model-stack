import argparse
import subprocess
import time
import os
import itertools
import json
import atexit
from typing import List, Dict, Any

_slurm_job_id = os.environ.get("SLURM_JOB_ID")
_cancel_on_exit = True # Flag to prevent double cancellation

def _cancel_slurm_job():
    """Attempts to cancel the Slurm job associated with this script."""
    global _cancel_on_exit
    if _slurm_job_id and _cancel_on_exit:
        print(f"Attempting to cancel Slurm job {_slurm_job_id} on exit...")
        subprocess.run(['scancel', _slurm_job_id], check=False)
        _cancel_on_exit = False # Prevent re-running if called from both atexit and finally

def run_inference_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs the inference script with the given configuration and returns timing results.
    """
    results = {"config": config, "success": False}
    base_command = [
        "torchrun",
        # nproc_per_node will be set based on attention type
        os.path.abspath(os.path.join(os.path.dirname(__file__), "inference.py")),
        "--architecture", config["architecture"],
        "--variant", config["variant"],
        "--model_path", config["model_path"],
        "--model_source", config["model_source"],
        "--tokenizer", config["tokenizer"],
        "--device_type", config["device_type"],
        "--max_new_tokens", str(config["max_new_tokens"]),
    ]
    # Pass the flag down, even if inference.py doesn't use it yet
    if config.get("print_response", False):
        base_command.append("--print_response")

    if config["dtype"]:
        base_command.extend(["--default_dtype", config["dtype"]])
    if not config["use_cache"]:
        base_command.append("--no_use_cache")

    env = os.environ.copy()
    # Add necessary env vars, potentially overriding existing ones
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Example, adjust as needed

    command = list(base_command) # Create a mutable copy

    # Configure based on attention type
    attn_type = config["attn_type"]
    if attn_type == "ring":
        command.insert(1, "--nproc_per_node=2") # Assuming 2 GPUs/processes for ring
        command.extend([
            "--distributed",
            "--distributed_strategy", "ring",
            "--attn_algorithm", "ring"
        ])
        # Pass the block size if specified in the config
        if config.get("ring_block_size"):
           command.extend(["--ring_block_size", str(config["ring_block_size"])])
    elif attn_type == "regular":
        command.insert(1, "--nproc_per_node=1") # Assuming 1 GPU/process for regular non-dist
        # No extra flags needed for standard attention
    else:
        raise ValueError(f"Unknown attention type: {attn_type}")

    print(f"\n--- Running Configuration ---")
    print(json.dumps(config, indent=2))
    print(f"Command: {' '.join(command)}")

    try:
        start_time = time.perf_counter()
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, # Don't raise exception on non-zero exit
            env=env,
        )
        end_time = time.perf_counter()

        results["duration_sec"] = end_time - start_time
        results["return_code"] = process.returncode
        results["stdout"] = process.stdout
        results["stderr"] = process.stderr

        if process.returncode == 0:
            results["success"] = True
            results["time_per_token_ms"] = (results["duration_sec"] * 1000) / config["max_new_tokens"]
            print(f"Success! Duration: {results['duration_sec']:.3f}s")
            print(f"Time per token: {results['time_per_token_ms']:.2f}ms")
        else:
            # Always print stdout/stderr on failure for debugging
            print("--- STDOUT ---")
            print(process.stdout)
            print(f"Error! Return Code: {process.returncode}")
            print("--- STDERR ---")
            print(process.stderr)

    except Exception as e:
        print(f"Failed to run subprocess: {e}")
        results["error"] = str(e)

    return results


if __name__ == "__main__":
    # Register the cancellation function to run on normal exit or unhandled exceptions
    atexit.register(_cancel_slurm_job)

    parser = argparse.ArgumentParser(description="Benchmark Ring vs Regular Attention Inference")
    # Add arguments matching your common setup needs
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--architecture", type=str, default="llama", help="Model architecture")
    parser.add_argument("--variant", type=str, default="7b", help="Model variant")
    parser.add_argument("--device_type", type=str, default="cuda", help="Device (cuda, cpu, mps)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Data type")
    parser.add_argument("--use_cache", action="store_true", help="Enable KV caching (default: False)")
    parser.add_argument("--attn_types", nargs='+', default=["ring", "regular"], help="Attention types to benchmark ('ring', 'regular')")
    parser.add_argument("--max_new_tokens", nargs='+', type=int, default=[256], help="List of max_new_tokens values to test")
    parser.add_argument("--print_response", action="store_true", help="Print the generated response captured from inference.py stdout")
    parser.add_argument("--ring_block_size", type=int, default=4096, help="Block size for Ring Attention runs")
    # Add other parameters you want to vary, e.g., context lengths (would require prompt generation)
    # parser.add_argument("--context_lengths", nargs='+', type=int, default=[512], help="List of context lengths to test")

    args = parser.parse_args()

    try:
        benchmark_configs = []
        # Generate configurations to test
        for attn_type, tokens in itertools.product(args.attn_types, args.max_new_tokens):
            config = {
                "architecture": args.architecture,
                "variant": args.variant,
                "model_path": os.path.abspath(args.model_path),
                "model_source": "hf", # Assuming HF source, adjust if needed
                "tokenizer": os.path.abspath(args.tokenizer),
                "device_type": args.device_type,
                "dtype": args.dtype,
                "use_cache": args.use_cache,
                "attn_type": attn_type,
                "max_new_tokens": tokens,
                "print_response": args.print_response, # Store the flag in the config
                "ring_block_size": args.ring_block_size if attn_type == "ring" else None, # Only relevant for ring
                # Add other varying params here, e.g. "context_length": length
            }
            benchmark_configs.append(config)

        all_results = []
        for config in benchmark_configs:
            result = run_inference_benchmark(config)
            all_results.append(result)

        print("\n--- Benchmark Summary ---")
        # TODO: Add more sophisticated reporting here (e.g., table, save to file)
        for result in all_results:
            print(f"Config: {result['config']}")
            if result["success"]:
                # If requested, print the captured stdout which contains the response
                if result['config'].get('print_response'):
                    print("--- Generated Response (stdout) ---")
                    print(result.get('stdout', 'N/A').strip()) # Use strip() to remove extra whitespace
                print(f"  Time/Token: {result.get('time_per_token_ms', 'N/A'):.2f} ms")
            else:
                print(f"  FAILED (Code: {result.get('return_code', 'N/A')}, Error: {result.get('error', 'See logs')})")
    finally:
        # Ensure cancellation is attempted even if atexit fails or script is killed abruptly
        _cancel_slurm_job()