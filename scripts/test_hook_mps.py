import argparse
import torch
import sys

# Make sure the FMS library is importable
# You might need to adjust this based on your PYTHONPATH or if you installed FMS
try:
    from fms.models import get_model
    from fms.utils import tokenizers
except ImportError:
    print("Error: Could not import FMS library. Make sure it's installed or in PYTHONPATH.")
    sys.exit(1)

# --- Simple Hook Function ---
def print_output_hook(module, input, output, layer_name=""):
    """A simple hook that prints info about the layer's output."""
    try:
        # Try to get the main tensor output, handling tuples/lists
        output_tensor = None
        if isinstance(output, torch.Tensor):
            output_tensor = output
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            output_tensor = output[0] # Assume first tensor is the main one

        if output_tensor is not None:
            print(f"--- Hook fired for: {layer_name} ---")
            print(f"  Output tensor shape: {output_tensor.shape}")
            print(f"  Output tensor device: {output_tensor.device}")
            print(f"  Output tensor dtype: {output_tensor.dtype}")
            # Calculate some stats (use float() to avoid potential dtype issues on MPS)
            try:
                 mean_val = output_tensor.float().mean().item()
                 max_val = output_tensor.float().max().item()
                 print(f"  Output tensor mean: {mean_val:.4f}")
                 print(f"  Output tensor max: {max_val:.4f}")
            except Exception as e:
                 print(f"  Could not calculate stats: {e}")
            print("-" * (len(layer_name) + 20))
        else:
            print(f"--- Hook fired for: {layer_name} ---")
            print(f"  Output type: {type(output)} (Could not extract single tensor)")
            print("-" * (len(layer_name) + 20))

    except Exception as e:
        print(f"Error in hook for {layer_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyTorch hooks on MPS with LLaMA")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights (HF format)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="Hello world!", help="Input prompt")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("mps")
    print(f"Using device: {device}")

    # --- Load Model and Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    print("Loading model...")
    # Load model initially on CPU or Meta to avoid potential MPS OOM during load, then move
    model = get_model("llama", "7b", model_path=args.model_path, source="hf", device_type="mps")
    model.to(device) # Move model to MPS
    model.eval()
    print("Model loaded.")

    # --- Register Hooks ---
    hook_handles = []
    try:
        # Adjust layer path if needed (e.g., model.base_model.layers[0])
        target_layer_0 = model.layers[0] if hasattr(model, 'layers') else model.base_model.layers[0]
        hook_handles.append(target_layer_0.attn.register_forward_hook(lambda m, i, o: print_output_hook(m, i, o, layer_name="Layer 0 Attention")))
        hook_handles.append(target_layer_0.ff_sub_layer.register_forward_hook(lambda m, i, o: print_output_hook(m, i, o, layer_name="Layer 0 MLP")))
        print("Hooks registered.")
    except Exception as e:
        print(f"Error registering hooks: {e}")
        sys.exit(1)

    # --- Prepare Input ---
    print(f"Tokenizing prompt: '{args.prompt}'")
    prompt_tokens = tokenizer.tokenize(args.prompt)
    encoded_prompt = tokenizer.convert_tokens_to_ids(prompt_tokens)
    if tokenizer.bos_token_id is not None: encoded_prompt.insert(0, tokenizer.bos_token_id)
    input_ids = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

    # --- Run Forward Pass ---
    print("Running forward pass...")
    with torch.no_grad():
        try:
            output = model(input_ids, use_cache=False) # Keep it simple, no cache or position_ids needed for this test
            print("Forward pass completed.")
            # Optional: print output logits info
            # print(f"Output logits shape: {output.shape}")
        except Exception as e:
            print(f"Error during forward pass: {e}")

    # --- Cleanup ---
    print("Removing hooks...")
    for handle in hook_handles:
        handle.remove()
    print("Done.")