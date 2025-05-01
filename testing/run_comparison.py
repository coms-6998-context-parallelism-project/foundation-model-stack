import argparse
import torch
import sys
import time
import os
from torch import distributed as dist

# Make sure the FMS library is importable
try:
    from fms.models import get_model
    from fms.utils import tokenizers, generation
except ImportError:
    print("Error: Could not import FMS library. Make sure it's installed (e.g. pip install -e .) or in PYTHONPATH.")
    sys.exit(1)

def load_model_for_config(args, attn_algo, device):
    """Loads the model with a specific attention algorithm configuration."""
    print(f"Loading model with attention: {attn_algo} on device: {device}")

    distr_strat = None
    group = None
    is_distributed = dist.is_available() and dist.is_initialized()

    if attn_algo == "ring":
        if is_distributed and args.device_type == 'cuda':
            group = dist.group.WORLD
            distr_strat = "ring"
            print(f"  Using distributed strategy: {distr_strat}, group size: {dist.get_world_size(group)}")
        else:
            print(f"  Warning: Ring attention requested but not running distributed on CUDA. Will likely fallback or fail.")
            # Allow fallback to standard attention if ring isn't possible
            attn_algo = None # Or 'sdpa' if that's the desired fallback
            distr_strat = None
    elif attn_algo == "sdpa":
        print(f"  Using SDPA attention.")
        # attn_algo = 'sdpa' # Explicitly set if needed by get_model or model.forward
    else: # Default or other algorithms
        print(f"  Using default attention mechanism.")
        attn_algo = None # Let the model use its default

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, None)

    # Load model (potentially on CPU first if large, then move)
    model = get_model(
        args.architecture,
        args.variant,
        model_path=args.model_path,
        source=args.model_source,
        device_type="cpu", # Load weights on CPU first
        distributed_strategy=distr_strat, # Pass strategy if applicable
        group=group,
        data_type=torch_dtype,
        # Pass attn_algorithm to get_model if it influences loading/structure
        # attn_algorithm=attn_algo, # This might depend on get_model implementation
    )

    # Configure attention post-load if necessary (highly model-dependent)
    # Example: if model.config controls it
    # if hasattr(model.config, 'attn_implementation'):
    #    model.config.attn_implementation = attn_algo if attn_algo else 'eager' # Or appropriate default

    model.to(device) # Move the entire model to the target device
    model.eval()
    print("Model loaded successfully.")
    return model, attn_algo # Return potentially modified attn_algo

def run_generate(model, tokenizer, prompt_ids, device, max_new_tokens, attn_algo_for_fwd):
    """Runs generation and returns the generated IDs."""
    print(f"Generating {max_new_tokens} tokens...")
    start_time = time.perf_counter()

    # Ensure prompt_ids is on the correct device
    prompt_ids = prompt_ids.to(device)

    # Use the fms generation utility
    generated_ids = generation.generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        use_cache=True, # Typically want cache for generation
        attn_algorithm=attn_algo_for_fwd # Pass the algorithm to forward
    )

    end_time = time.perf_counter()
    duration = end_time - start_time
    num_generated = generated_ids.shape[1] - prompt_ids.shape[1]
    print(f"Generation complete. Generated {num_generated} tokens in {duration:.2f}s")
    if num_generated > 0:
        print(f"Throughput: {num_generated / duration:.2f} tokens/sec")

    # Return only the generated part (excluding prompt) on CPU
    return generated_ids[0, prompt_ids.shape[1]:].cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare LLaMA generation with different attention mechanisms")
    # Model args
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--architecture", type=str, default="llama", help="Model architecture")
    parser.add_argument("--variant", type=str, default="7b", help="Model variant")
    parser.add_argument("--model_source", type=str, default="hf", help="Source format (hf, meta)")
    # Execution args
    parser.add_argument("--device_type", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"], help="Data type")
    # Generation args
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Number of new tokens to generate")
    # Comparison args
    parser.add_argument("--compare", type=str, nargs=2, required=True, metavar=('BASELINE_ALGO', 'TEST_ALGO'),
                        help="Two attention algorithms to compare (e.g., sdpa ring)")

    args = parser.parse_args()
    baseline_algo_req, test_algo_req = args.compare

    # --- Distributed Setup (if applicable, e.g., for Ring on CUDA) ---
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    is_distributed = world_size > 1

    if is_distributed and args.device_type == 'cuda':
        if not dist.is_initialized():
            print(f"Initializing distributed process group (backend: nccl, world_size: {world_size})")
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"Running distributed on rank {local_rank}/{world_size} with device {device}")
    elif args.device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device_type == 'mps' and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if args.device_type != 'cpu':
            print(f"Warning: Requested device {args.device_type} not available, using CPU.")
        device = torch.device("cpu")
        args.device_type = 'cpu' # Update args to reflect reality

    # Only rank 0 should print most messages and do the final comparison
    is_rank_0 = local_rank == 0

    # --- Tokenizer and Prompt ---
    if is_rank_0: print("Loading tokenizer...")
    tokenizer = tokenizers.get_tokenizer(args.tokenizer)
    if is_rank_0: print(f"Tokenizing prompt: '{args.prompt}'")
    prompt_tokens = tokenizer.tokenize(args.prompt)
    encoded_prompt = tokenizer.convert_tokens_to_ids(prompt_tokens)
    if tokenizer.bos_token_id is not None and (not encoded_prompt or encoded_prompt[0] != tokenizer.bos_token_id):
        encoded_prompt.insert(0, tokenizer.bos_token_id)
    # Keep prompt IDs on CPU for now, move to device in run_generate
    prompt_ids_cpu = torch.tensor([encoded_prompt], dtype=torch.long, device='cpu')

    # --- Run Baseline ---
    if is_rank_0: print(f"\n--- Running Baseline: {baseline_algo_req} ---")
    model_baseline, baseline_algo_used = load_model_for_config(args, baseline_algo_req, device)
    baseline_generated_ids = run_generate(model_baseline, tokenizer, prompt_ids_cpu, device, args.max_new_tokens, baseline_algo_used)
    del model_baseline # Free memory
    if args.device_type == 'cuda': torch.cuda.empty_cache()

    # --- Run Test ---
    if is_rank_0: print(f"\n--- Running Test: {test_algo_req} ---")
    model_test, test_algo_used = load_model_for_config(args, test_algo_req, device)
    test_generated_ids = run_generate(model_test, tokenizer, prompt_ids_cpu, device, args.max_new_tokens, test_algo_used)
    del model_test # Free memory
    if args.device_type == 'cuda': torch.cuda.empty_cache()

    # --- Comparison (only on Rank 0 if distributed) ---
    if is_rank_0:
        print("\n--- Comparison Results ---")
        print(f"Baseline Algorithm Used: {baseline_algo_used if baseline_algo_used else 'default'}")
        print(f"Test Algorithm Used: {test_algo_used if test_algo_used else 'default'}")

        len_baseline = baseline_generated_ids.shape[0]
        len_test = test_generated_ids.shape[0]
        min_len = min(len_baseline, len_test)

        print(f"Baseline generated {len_baseline} tokens.")
        print(f"Test generated {len_test} tokens.")

        if min_len == 0:
            print("One or both runs generated zero tokens. Cannot compare sequence.")
        else:
            match = torch.equal(baseline_generated_ids[:min_len], test_generated_ids[:min_len])
            if match and len_baseline == len_test:
                print("✅ SUCCESS: Generated sequences are identical.")
            elif match and len_baseline != len_test:
                print(f"⚠️ WARNING: Sequences match up to minimum length ({min_len}), but final lengths differ.")
            else:
                # Find first mismatch
                mismatch_idx = -1
                for i in range(min_len):
                    if baseline_generated_ids[i] != test_generated_ids[i]:
                        mismatch_idx = i
                        break
                print(f"❌ FAILURE: Sequences differ.")
                if mismatch_idx != -1:
                    print(f"  First mismatch at token index {mismatch_idx} (0-based):")
                    print(f"    Baseline: {baseline_generated_ids[mismatch_idx].item()} ('{tokenizer.convert_ids_to_tokens([baseline_generated_ids[mismatch_idx].item()])[0]}')")
                    print(f"    Test:     {test_generated_ids[mismatch_idx].item()} ('{tokenizer.convert_ids_to_tokens([test_generated_ids[mismatch_idx].item()])[0]}')")
                else: # Should only happen if one sequence is prefix of other but match=False (logic error?)
                     print("  Sequences differ but mismatch index not found (unexpected).")

        # Optionally print decoded strings
        print("\n--- Decoded Baseline ---")
        baseline_tokens = tokenizer.convert_ids_to_tokens(baseline_generated_ids.tolist())
        print(tokenizer.convert_tokens_to_string(baseline_tokens))
        print("\n--- Decoded Test ---")
        test_tokens = tokenizer.convert_ids_to_tokens(test_generated_ids.tolist())
        print(tokenizer.convert_tokens_to_string(test_tokens))
        print("\n--- End Comparison ---")

    # Cleanup distributed group if we initialized it here
    if is_distributed and dist.is_initialized() and world_size > 1:
         dist.destroy_process_group()
         print(f"Rank {local_rank} destroyed process group.")