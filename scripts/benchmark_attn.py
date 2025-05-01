import argparse
import sys
import time
import os
import atexit
import torch
from collections import defaultdict
import functools # Needed for passing args to hooks easily

# Assuming these are needed based on the new logic
from fms.models import get_model
from fms.utils import generation, tokenizers

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
import subprocess # Keep for _cancel_slurm_job

# Define tolerance for comparison (can be overridden by args)
ATOL = 1e-5
RTOL = 1e-3

# Global storage for baseline outputs and errors
baseline_outputs = defaultdict(lambda: defaultdict(dict))
comparison_errors = []
current_token_idx = 0 # Global to track generation step within hooks

# --- Hook Function Factory ---
def get_comparison_hook(layer_idx, output_name, is_baseline_run, baseline_attn_type, test_attn_type):
    """Creates a forward hook to capture (baseline) or compare (test) outputs."""
    global baseline_outputs, comparison_errors, current_token_idx, ATOL, RTOL

    def hook(module, input, output):
        # We usually care about the main hidden_state tensor
        # Output might be a tuple (hidden_states, cache, ...) - adapt if needed
        try:
            output_tensor = output[0] if isinstance(output, tuple) else output
            output_tensor = output_tensor.detach() # Detach from graph
        except Exception as e:
            print(f"[Hook Error] Could not extract tensor from output at Layer {layer_idx}, Output '{output_name}'. Error: {e}", file=sys.stderr)
            print(f"Output type: {type(output)}", file=sys.stderr)
            # Add error and potentially stop? For now, just log it.
            comparison_errors.append(f"Hook Error: Could not extract tensor at Layer {layer_idx}, Output '{output_name}'")
            return # Cannot proceed with this hook instance

        if is_baseline_run:
            # Store the baseline output (move to CPU to save GPU memory)
            baseline_outputs[current_token_idx][layer_idx][output_name] = output_tensor.clone().cpu()
            # print(f"[Baseline] Stored: Token {current_token_idx}, Layer {layer_idx}, Output '{output_name}', Shape {output_tensor.shape}") # Debug print
        else:
            # Compare with the baseline output
            if current_token_idx in baseline_outputs and \
               layer_idx in baseline_outputs[current_token_idx] and \
               output_name in baseline_outputs[current_token_idx][layer_idx]:

                baseline_tensor = baseline_outputs[current_token_idx][layer_idx][output_name].to(output_tensor.device, dtype=output_tensor.dtype)
                # print(f"[Compare] Comparing: Token {current_token_idx}, Layer {layer_idx}, Output '{output_name}', Shape {output_tensor.shape} vs {baseline_tensor.shape}") # Debug print

                # Perform comparison
                are_close = torch.allclose(output_tensor, baseline_tensor, atol=ATOL, rtol=RTOL)

                if not are_close:
                    diff = torch.abs(output_tensor - baseline_tensor)
                    max_diff = torch.max(diff).item()
                    mean_diff = torch.mean(diff).item()
                    error_msg = (
                        f"--- MISMATCH DETECTED ---\n"
                        f"  Baseline Attention: '{baseline_attn_type}'\n"
                        f"  Test Attention: '{test_attn_type}'\n"
                        f"  Token Index (0-based): {current_token_idx}\n"
                        f"  Layer Index (0-based): {layer_idx}\n"
                        f"  Output Name: '{output_name}'\n"
                        f"  Shapes: Test={output_tensor.shape}, Baseline={baseline_tensor.shape}\n"
                        f"  Max Absolute Difference: {max_diff:.6e}\n"
                        f"  Mean Absolute Difference: {mean_diff:.6e}\n"
                        f"  Comparison failed with atol={ATOL}, rtol={RTOL}"
                    )
                    comparison_errors.append(error_msg)
                    # Optional: Add more debug info like printing tensor slices
                    # print("Current Tensor (sample):", output_tensor.flatten()[:10])
                    # print("Baseline Tensor (sample):", baseline_tensor.flatten()[:10])
            else:
                # This indicates a potential logic error or different generation lengths
                error_msg = (
                    f"--- COMPARISON LOGIC ERROR ---\n"
                    f"  Baseline output missing for comparison!\n"
                    f"  Token Index: {current_token_idx}\n"
                    f"  Layer Index: {layer_idx}\n"
                    f"  Output Name: '{output_name}'\n"
                    f"  This might happen if generation lengths differ unexpectedly."
                )
                comparison_errors.append(error_msg)

    return hook

# --- Helper functions (assuming these exist or are adapted) ---
def load_tokenizer(tokenizer_path):
    """Loads the tokenizer."""
    # Replace with your actual tokenizer loading logic
    return tokenizers.get_tokenizer(tokenizer_path)

def load_model(args, attn_type):
    """Loads the model with the specified attention type."""
    # Replace with your actual model loading logic using get_model
    # Ensure it passes the correct attn_algorithm based on attn_type
    # and handles distributed setup if needed (especially for 'ring')
    print(f"Loading model with attention type: {attn_type}")

    # Determine distributed strategy based on attn_type
    # This assumes 'ring' needs distributed setup, 'sdpa' does not. Adjust as needed.
    distr_strat = None
    attn_algo = None
    group = None
    if attn_type == "ring":
        if not dist.is_initialized():
             print("Initializing process group for Ring Attention...")
             dist.init_process_group() # Basic initialization, might need more config
        group = dist.group.WORLD
        distr_strat = "ring"
        attn_algo = "ring"
        print(f"Using distributed strategy: {distr_strat}, group size: {dist.get_world_size(group)}")
    elif attn_type == "sdpa":
        # Assuming sdpa runs on a single process/GPU
        distr_strat = None # Or NoOpStrategy if explicitly needed
        attn_algo = "sdpa" # Or None if sdpa is the default
        print(f"Using non-distributed setup for {attn_type}")
    else:
        raise ValueError(f"Unsupported attention type for loading: {attn_type}")

    # Map dtype string to torch dtype
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(args.dtype, None)

    model = get_model(
        args.architecture,
        args.variant,
        model_path=args.model_path,
        source=args.model_source,
        device_type=args.device_type,
        distributed_strategy=distr_strat,
        group=group,
        data_type=torch_dtype,
        # Pass attn_algorithm to get_model if it influences model structure/loading
        # attn_algorithm=attn_algo, # This might not be needed if handled post-load
    )
    # Crucially, ensure the loaded model uses the correct attention implementation.
    # This might involve setting a config flag or modifying the model post-load
    # if get_model doesn't handle it directly via an argument.
    # Example (conceptual): model.config.attn_implementation = attn_type

    return model

# --- Main Comparison Function ---
def run_comparison_benchmark(args):
    global baseline_outputs, comparison_errors, current_token_idx, ATOL, RTOL

    # Set tolerance from args
    ATOL = args.atol
    RTOL = args.rtol

    if len(args.attn_types_to_compare) != 2:
        print("Error: --attn_types_to_compare requires exactly two attention types (baseline first, test second).", file=sys.stderr)
        sys.exit(1)

    baseline_attn_type, test_attn_type = args.attn_types_to_compare
    print(f"Starting comparison: Baseline='{baseline_attn_type}', Test='{test_attn_type}'")
    print(f"Comparison tolerance: atol={ATOL}, rtol={RTOL}")

    # Common setup
    tokenizer = load_tokenizer(args.tokenizer)
    # Use a fixed prompt for consistency
    prompt = args.prompt if args.prompt else "Once upon a time"
    print(f"Using prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device_type)
    input_ids = inputs.input_ids
    max_gen_len = args.max_new_tokens

    # --- 1. Baseline Run ---
    print(f"\n--- Running Baseline: {baseline_attn_type} ---")
    # Note: Loading might need distributed setup depending on attn_type
    model_baseline = load_model(args, baseline_attn_type)
    model_baseline.eval()

    # Register hooks for baseline run
    hooks_baseline = []
    try:
        num_layers = model_baseline.config.nlayers # Adapt if config structure differs
    except AttributeError:
        print("Error: Could not determine number of layers from model_baseline.config.nlayers", file=sys.stderr)
        sys.exit(1)

    print(f"Registering hooks for {num_layers} layers...")
    for i in range(num_layers):
        try:
            # Adjust path based on actual model structure (e.g., model.layers or model.base_model.layers)
            layer = model_baseline.base_model.layers[i] if hasattr(model_baseline, 'base_model') else model_baseline.layers[i]
            # Hook after attention module
            hooks_baseline.append(
                layer.attn.register_forward_hook( # Assuming attn is the attribute name
                    get_comparison_hook(i, "attn_output", True, baseline_attn_type, test_attn_type)
                )
            )
            # Hook after MLP module
            hooks_baseline.append(
                layer.ff_sub_layer.register_forward_hook( # Assuming ff_sub_layer is the attribute name
                     get_comparison_hook(i, "mlp_output", True, baseline_attn_type, test_attn_type)
                )
            )
            # Hook after the full layer (including residuals, norms)
            hooks_baseline.append(
                layer.register_forward_hook(
                     get_comparison_hook(i, "layer_output", True, baseline_attn_type, test_attn_type)
                )
            )
        except Exception as e:
            print(f"Error registering hook for layer {i}: {e}", file=sys.stderr)
            # Clean up already registered hooks before exiting
            for handle in hooks_baseline: handle.remove()
            sys.exit(1)

    # Manual generation loop for baseline
    print(f"Generating baseline tokens (max {max_gen_len})...")
    generated_ids_baseline = input_ids
    past_key_values = None
    baseline_outputs.clear() # Ensure clean state
    comparison_errors.clear() # Ensure clean state
    start_time_baseline = time.perf_counter()
    with torch.no_grad():
        for step in range(max_gen_len):
            current_token_idx = step # Set global index for hooks
            # Prepare inputs for the next step
            # Note: Using use_cache=True is important for performance and correctness
            # The prepare_inputs_for_generation method might not exist, adapt if needed
            current_input_ids = generated_ids_baseline[:, -1:] if use_cache and past_key_values else generated_ids_baseline

            # Forward pass - hooks will capture data here
            # Pass attn_algorithm explicitly if needed by the model's forward
            outputs = model_baseline(
                current_input_ids,
                past_key_value_states=past_key_values,
                use_cache=args.use_cache,
                attn_algorithm=baseline_attn_type if baseline_attn_type == 'ring' else None # Pass 'ring' if baseline is ring
            )

            # Check for unexpected errors during baseline hook execution
            if comparison_errors:
                 print("\n!!! UNEXPECTED ERROR DURING BASELINE HOOK EXECUTION !!!", file=sys.stderr)
                 for error in comparison_errors: print(error, file=sys.stderr)
                 # Clean up hooks
                 for handle in hooks_baseline: handle.remove()
                 sys.exit(1)

            # Process outputs (assuming standard output format)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if args.use_cache:
                past_key_values = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None

            # Get the next token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids_baseline = torch.cat([generated_ids_baseline, next_token], dim=-1)

            # Check for EOS
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                print(f"Baseline EOS token reached at step {step}.")
                break
    end_time_baseline = time.perf_counter()

    # Remove baseline hooks
    for handle in hooks_baseline:
        handle.remove()
    print("Baseline run complete.")
    baseline_duration = end_time_baseline - start_time_baseline
    baseline_tokens_generated = generated_ids_baseline.shape[1] - input_ids.shape[1]
    print(f"Baseline Duration: {baseline_duration:.3f}s")
    if baseline_tokens_generated > 0:
        print(f"Baseline Time per token: {(baseline_duration * 1000) / baseline_tokens_generated:.2f}ms")

    baseline_response = tokenizer.decode(generated_ids_baseline[0], skip_special_tokens=True)
    print(f"Baseline Response Length: {baseline_tokens_generated} tokens")
    if args.print_response: print(f"Baseline Response:\n{baseline_response}")

    # Clear baseline model from memory
    del model_baseline
    del past_key_values
    torch.cuda.empty_cache()

    # --- 2. Test Run ---
    print(f"\n--- Running Test: {test_attn_type} ---")
    model_test = load_model(args, test_attn_type) # Load model with the test attention type
    model_test.eval()

    # Register hooks for test run (comparison mode)
    hooks_test = []
    try:
        num_layers_test = model_test.config.nlayers
    except AttributeError:
        print("Error: Could not determine number of layers from model_test.config.nlayers", file=sys.stderr)
        sys.exit(1)

    if num_layers != num_layers_test:
         print(f"Error: Baseline ({num_layers}) and test ({num_layers_test}) models have different numbers of layers!", file=sys.stderr)
         sys.exit(1)

    print(f"Registering comparison hooks for {num_layers} layers...")
    for i in range(num_layers):
        try:
            # Adjust path based on actual model structure
            layer = model_test.base_model.layers[i] if hasattr(model_test, 'base_model') else model_test.layers[i]
            hooks_test.append(layer.attn.register_forward_hook(get_comparison_hook(i, "attn_output", False, baseline_attn_type, test_attn_type)))
            hooks_test.append(layer.ff_sub_layer.register_forward_hook(get_comparison_hook(i, "mlp_output", False, baseline_attn_type, test_attn_type)))
            hooks_test.append(layer.register_forward_hook(get_comparison_hook(i, "layer_output", False, baseline_attn_type, test_attn_type)))
        except Exception as e:
            print(f"Error registering hook for layer {i}: {e}", file=sys.stderr)
            for handle in hooks_test: handle.remove()
            sys.exit(1)

    # Manual generation loop for test run
    # Generate only as many tokens as the baseline did
    num_tokens_to_generate = baseline_tokens_generated
    print(f"Generating test tokens and comparing (up to {num_tokens_to_generate} tokens)...")
    generated_ids_test = input_ids
    past_key_values = None
    comparison_errors.clear() # Reset errors for test run
    start_time_test = time.perf_counter()

    with torch.no_grad():
        for step in range(num_tokens_to_generate):
            current_token_idx = step # Set global index for hooks
            current_input_ids = generated_ids_test[:, -1:] if use_cache and past_key_values else generated_ids_test

            # Forward pass - hooks will compare data here
            # Pass attn_algorithm explicitly if needed
            outputs = model_test(
                current_input_ids,
                past_key_value_states=past_key_values,
                use_cache=args.use_cache,
                attn_algorithm=test_attn_type if test_attn_type == 'ring' else None # Pass 'ring' if test is ring
            )

            # Check for comparison errors *after* the forward pass for this token
            if comparison_errors:
                print("\n!!! COMPARISON FAILED !!!", file=sys.stderr)
                # Print all accumulated errors
                for error in comparison_errors:
                    print(error, file=sys.stderr)
                # Clean up test hooks before exiting
                for handle in hooks_test: handle.remove()
                sys.exit(1) # Terminate on first detected error

            # Process outputs
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if args.use_cache:
                past_key_values = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None

            # Get the next token
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_ids_test = torch.cat([generated_ids_test, next_token], dim=-1)

            # Optional: Check if generated token matches baseline token at this step
            baseline_next_token = generated_ids_baseline[:, input_ids.shape[1] + step]
            if next_token.item() != baseline_next_token.item():
                 print(f"\nWarning: Generated token mismatch at step {step}!")
                 print(f"  Test generated: {next_token.item()} ('{tokenizer.decode(next_token[0])}')")
                 print(f"  Baseline generated: {baseline_next_token.item()} ('{tokenizer.decode(baseline_next_token)}')")
                 # This is often expected if intermediate values diverge even slightly

            # Check for EOS (mainly to stop unnecessary generation if test finishes early)
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                print(f"Test EOS token reached at step {step}.")
                # Check if baseline also finished here or earlier
                if step < num_tokens_to_generate -1 :
                     print(f"Warning: Test finished earlier than baseline (baseline length: {num_tokens_to_generate} new tokens).")
                break
    end_time_test = time.perf_counter()

    # Remove test hooks
    for handle in hooks_test:
        handle.remove()

    # Final check if loop completed without errors
    if not comparison_errors:
        print("\n--- Comparison Successful ---")
        print(f"All monitored intermediate outputs for {num_tokens_to_generate} generated tokens matched between")
        print(f"'{baseline_attn_type}' and '{test_attn_type}' within tolerance (atol={ATOL}, rtol={RTOL}).")

        test_duration = end_time_test - start_time_test
        test_tokens_generated = generated_ids_test.shape[1] - input_ids.shape[1]
        print(f"Test Duration: {test_duration:.3f}s")
        if test_tokens_generated > 0:
             print(f"Test Time per token: {(test_duration * 1000) / test_tokens_generated:.2f}ms")

        test_response = tokenizer.decode(generated_ids_test[0], skip_special_tokens=True)
        print(f"Test Response Length: {test_tokens_generated} tokens")
        if args.print_response: print(f"Test Response:\n{test_response}")

        # Final check: are the full generated sequences identical?
        min_len = min(generated_ids_baseline.shape[1], generated_ids_test.shape[1])
        if torch.equal(generated_ids_baseline[:, :min_len], generated_ids_test[:, :min_len]):
            if generated_ids_baseline.shape[1] == generated_ids_test.shape[1]:
                 print("\nFinal generated sequences are identical.")
            else:
                 print(f"\nFinal generated sequences are identical up to minimum length ({min_len} total tokens), but lengths differ.")
        else:
            print("\nWarning: Final generated sequences differ even before potential length difference.")

    else:
         # This case should technically be caught within the loop, but as a safeguard:
         print("\n!!! COMPARISON FAILED (Detected after loop) !!!", file=sys.stderr)
         for error in comparison_errors: print(error, file=sys.stderr)
         sys.exit(1)

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
    parser.add_argument("--use_cache", action="store_true", default=True, help="Enable KV caching (default: True)") # Changed default
    parser.add_argument("--max_new_tokens", nargs='+', type=int, default=[256], help="List of max_new_tokens values to test")
    parser.add_argument("--print_response", action="store_true", help="Print the generated response captured from inference.py stdout")
    parser.add_argument("--ring_block_size", type=int, default=4096, help="Block size for Ring Attention runs")
    parser.add_argument("--debug_ring", action="store_true", help="Enable detailed ring attention debugging prints in inference.py")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for generation")
    parser.add_argument("--model_source", type=str, default="hf", help="Source format of the model weights (e.g., hf, meta)")

    # Add comparison-specific arguments
    parser.add_argument("--attn_types_to_compare", type=str, nargs=2, required=True,
                        metavar=('BASELINE_TYPE', 'TEST_TYPE'),
                        help="Two attention types to compare (e.g., sdpa ring). The first is the baseline, the second is tested against it.")
    parser.add_argument("--atol", type=float, default=ATOL,
                        help=f"Absolute tolerance for torch.allclose comparison (default: {ATOL})")
    parser.add_argument("--rtol", type=float, default=RTOL,
                        help=f"Relative tolerance for torch.allclose comparison (default: {RTOL})")

    args = parser.parse_args()

    # For simplicity in this script, we only use the first value if multiple max_new_tokens are provided.
    # The original script's loop over tokens is removed.
    args.max_new_tokens = args.max_new_tokens[0]

    try:
        run_comparison_benchmark(args)
    finally:
        # Ensure cancellation is attempted even if atexit fails or script is killed abruptly
        _cancel_slurm_job()