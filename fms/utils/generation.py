import logging
import time
from typing import Any, Callable, Iterable, List, MutableMapping, Optional, Tuple, Union
import os # For rank/world size if not passed

import torch
import torch.nn.functional as F
import torch.distributed as dist # Import torch.distributed

# Import RingAttentionStrategy and distributed utils
from fms.distributed.strategy import RingAttentionStrategy
from fms import distributed

from fms.modules.ssm import SSMCacheUnit


logger = logging.getLogger(__name__)


def pad_input_ids(
    input_ids_list: List[torch.Tensor],
    min_pad_length: int = 0,
    is_causal_mask=True,
) -> Tuple[torch.Tensor, MutableMapping[str, Any]]:
    """
    Convert a list of Tensors to a rectangular tensor. Return extra padding kwargs for the position_ids and mask, since
    this will be required to properly handle the rectangular tensor for certain models.

    Parameters
    ----------
    input_ids_list: List[torch.Tensor]
        a list of Tensors of varied length
    min_pad_length: int
        pad to a min length provided. If the min_pad_length is less than the largest input_ids in the input_ids_list,
        padding will be determined based on the largest length input_ids.

    Returns
    -------
    Tuple[torch.Tensor, MutableMapping[str, Any]]
        A rectangular 2d padded tensor and a mapping containing the mask and position_ids typically used in forward pass
        in fms models
        A mapping from mask to a 3d causal mask and from position_ids to a 2d rectangular position_ids tensor
    """
    max_len = max([min_pad_length] + [seq.size(0) for seq in input_ids_list])

    padded_input_ids_list = []
    mask_list = []
    position_ids_list = []
    for input_ids_i in input_ids_list:
        seq_len = input_ids_i.size(0)
        pads = torch.zeros(
            max_len - seq_len, dtype=torch.long, device=input_ids_i.device
        )
        non_pads = torch.ones(seq_len, dtype=torch.bool, device=input_ids_i.device)

        # Setting this to 0, however if 0 is the eos, we will end up truncating the output if using truncate_after_eos
        # once this workflow works for nested tensor, this can probably be removed
        padded_input_ids_list.append(torch.cat((pads, input_ids_i)))

        # computing this as it's lightweight but could potentially be skipped
        mask_list.append(torch.cat((pads.bool(), non_pads)))

        pos_ids_pads = pads
        pos_ids_seq = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids_i.device
        )
        position_ids_list.append(torch.cat((pos_ids_pads, pos_ids_seq)))

    input_ids = torch.stack(padded_input_ids_list)
    padding_kwargs = {}
    mask = torch.stack(mask_list)
    mask = mask.unsqueeze(-1) == mask.unsqueeze(-2)
    # this is a causal mask for generation
    if is_causal_mask:
        mask = mask.tril()
    mask = torch.where(mask.logical_not(), -torch.inf, 0.0)
    padding_kwargs["mask"] = mask

    position_ids = torch.stack(position_ids_list)
    padding_kwargs["position_ids"] = position_ids

    return input_ids, padding_kwargs


def __update_padding_kwargs(
    use_cache: bool, model_specific_kwargs: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Generic function to prepare any model specific keyword arguments"""
    # extend the attention mask
    mask = model_specific_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.zeros(mask.size(0), 1, 1, device=mask.device),
            ),
            dim=2,
        )
        model_specific_kwargs["mask"] = mask
        if torch._dynamo.config.dynamic_shapes:
            torch._dynamo.mark_dynamic(mask, 2)

    # extend the position_ids
    position_ids = model_specific_kwargs.get("position_ids", None)
    if position_ids is not None:
        if use_cache:
            position_ids = position_ids[:, -1:] + 1
        else:
            position_ids = torch.cat(
                (position_ids, position_ids[:, -1:] + 1),
                dim=1,
            )
        model_specific_kwargs["position_ids"] = position_ids
    return model_specific_kwargs


def _make_cache_contiguous(
    past_key_value_states: list[Union[Iterable[torch.Tensor], SSMCacheUnit]],
) -> list[Union[Iterable[torch.Tensor], SSMCacheUnit]]:
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    n_kv_s: list[Union[Iterable[torch.Tensor], SSMCacheUnit]] = []
    for layer_cache in past_key_value_states:
        if (
            isinstance(layer_cache, Iterable)
            and all(
                [
                    isinstance(cache_element, torch.Tensor)
                    for cache_element in layer_cache
                ]
            )
            and any(
                [not cache_element.is_contiguous() for cache_element in layer_cache]
            )
        ):
            n_kv_s.append(
                tuple(
                    [
                        cache_element.clone(
                            memory_format=torch.contiguous_format
                        ).detach()
                        for cache_element in layer_cache
                    ]
                )
            )
        else:
            n_kv_s.append(layer_cache)
    return n_kv_s


def _make_cache_dynamic(
    past_key_value_states: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    for layer in past_key_value_states:
        if isinstance(layer, Iterable):
            for tensor in layer:
                torch._dynamo.mark_dynamic(tensor, 2)
    return past_key_value_states


def generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 4096,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    eos_token_id: Optional[int] = None,
    timing: str = "",
    prepare_model_inputs_hook: Optional[
        Callable[
            [int, torch.Tensor, MutableMapping[str, Any]],
            Tuple[torch.Tensor, MutableMapping[str, Any]],
        ]
    ] = None,
    post_iteration_hook: Optional[
        Callable[
            [int, torch.Tensor, torch.Tensor, MutableMapping[str, Any]],
            Tuple[torch.Tensor, MutableMapping[str, Any]],
        ]
    ] = None,
    extra_kwargs: Optional[MutableMapping[str, Any]] = None,
    attn_algorithm: Optional[str] = None, # Add attn_algorithm parameter
    debug_ring: bool = False, # Add debug flag
    tokenizer = None, # Add tokenizer for debug prints
    **kwargs, # Keep accepting other kwargs
):
    """
    A trivial generate function that can be used for validation/testing in
    cases where HF is not available.
    We could add implementations for other types of generation, but this is
    enough for making sure a model is working.
    Does not implement beam search, but this can be added.

    Args:
        model: A function or nn.Module that takes a batch of input_ids and
            returns logits
        input_ids: a rectangular tensor of input_ids (batch x seq)
        max_seq_len: the sequence length of the model
        max_new_tokens: max tokens to generate
        temperature: temperature of softmax when sampling
        top_k: only search among top k tokens
        do_sample: multinomial sampling. False for greedy.
        num_beams: TODO: support beam search
        use_cache: requires that the model accept use_cache and
            past_key_value_states args in forward method.
        contiguous_cache: ensures the cache is contiguous in device memory
        eos_token_id: the optional token id representing the end of sequence
        timing: whether to measure timings: "per-token" for each token generation time,
            "e2e" for full generation loop. Both options make `generate` return a tuple
            with the following information:
            - "per-token": Array with `max_new_tokens` time measurements (in s)
            - "e2e": Array with a single e2e generation loop time measurement (in s)
        prepare_model_inputs_hook: a function that will get called immediately before model forward.
            It must have the following signature: f(int generate_iteration, Tensor input_ids, Dict kwargs) ->
            Tuple[Tensor input_ids, Dict kwargs]. If it is defined, will replace input_ids
            and kwargs to next model forward based on the contents of the function.
        post_iteration_hook: a function that will get called after each iteration.
            It must have the following signature: f(int token_position, Tensor logits, Tensor next_val, Dict kwargs) ->
            Tuple[Tensor next_val, Dict kwargs]. If it is defined, will replace next_val
            and kwargs based on the contents of the function.
        extra_kwargs: an optional mapping of additional kwargs to pass to the model.
            For example: if extra_kwargs contains position_ids and mask keys, these
            model parameters will be updated as-appropriate for each token generated.
        attn_algorithm: Optional string specifying the attention algorithm to use (e.g., 'ring').
        debug_ring: bool, enable detailed debugging prints.
        tokenizer: Optional tokenizer object, needed for some debug prints.
        **kwargs: Additional keyword arguments to pass to the model's forward function.
    """
    # Get rank/world_size for debug prints if distributed
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    device = input_ids.device # Get device for memory print

    # Debug 1: Print rank and world size at the start of generation
    if debug_ring and is_distributed:
        print(f"[RANK {rank}] Initialized. World size: {dist.get_world_size()}", flush=True)

    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")

    # Rename kwargs to model_kwargs to avoid conflict with function kwargs
    model_kwargs: MutableMapping[str, Any] = dict()
    if extra_kwargs is not None:
        model_kwargs.update(extra_kwargs)

    if isinstance(input_ids, torch.Tensor):
        is_batch = len(input_ids.shape) > 1
        # our model requires batch dimension
        if not is_batch:
            input_ids = input_ids.unsqueeze(0)
        # Debug 2: Print input prompt shape and contents at rank 0
        if debug_ring and rank == 0:
            print(f"[RANK 0] Initial input_ids: {input_ids.tolist()}", flush=True)
        # Debug 3: Verify all ranks have the same tokenizer output (hash check)
        if debug_ring:
            print(f"[RANK {rank}] input_ids hash: {hash(str(input_ids.tolist()))}", flush=True)
    else:
        # Debug 2: Print input prompt shape and contents at rank 0 (list case)
        if debug_ring and rank == 0 and tokenizer:
             print(f"[RANK 0] Initial input_ids list: {[tokenizer.decode(t) for t in input_ids]}", flush=True)
        # Debug 3: Verify all ranks have the same tokenizer output (hash check list case)
        if debug_ring:
            print(f"[RANK {rank}] Initial input_ids hash: {hash(str(input_ids))}", flush=True)
        # Debug 14: Check model is in eval mode (requires model access)
        if debug_ring and isinstance(model, torch.nn.Module):
            print(f"[RANK {rank}] Model eval mode: {not model.training}", flush=True)
            # Debug 11: Log attention type being used at init (requires model access)
            # Debug 12: Dump selected attention module class name (requires model access)
            # Debug 13: Print per-layer rotary embedding device and shape (requires model access)
            # These require more specific model structure knowledge, adding placeholders
            # print(f"[RANK {rank}] Attention type: {getattr(model.config, 'attn_type', 'N/A')}", flush=True)
            # if hasattr(model, 'layers') and len(model.layers) > 0:
            #     print(f"[RANK {rank}] Using attention class: {type(model.layers[0].attn).__name__}", flush=True)
            #     if hasattr(model.layers[0].attn, 'rotary_emb'):
            #         print(f"[RANK {rank}] Layer 0 rotary_emb.inv_freq: {model.layers[0].attn.rotary_emb.inv_freq.shape} on {model.layers[0].attn.rotary_emb.inv_freq.device}", flush=True)
        raise TypeError("input_ids must be one of Tensor or List")

    eos_found = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )

    result = input_ids
    next_input = input_ids
    model_kwargs["past_key_value_states"] = None
    model_kwargs["use_cache"] = use_cache

    prompt_length = input_ids.shape[1]

    if timing != "":
        times: List[float] = []
        start_time = time.time()

    for i in range(max_new_tokens):
        # Debug 15: Measure and print time per iteration
        iter_start_time = time.time()

        # --- Ring Attention Length Check ---
        # Check if the next token would exceed the max length allowed by Ring Attention's fixed blocks
        strategy = getattr(model, "distributed_strategy", None)
        if isinstance(strategy, RingAttentionStrategy):
            # The max length the ring mechanism can handle due to fixed block sizes
            max_ring_len = strategy.world_size * strategy.block_size
            if debug_ring:
                print(f"[RANK {rank}] Iter {i} Current result len: {result.shape[1]}, Max ring len: {max_ring_len}", flush=True)
            # Check if the *current* length already meets or exceeds the limit.
            # The loop should break *before* trying to generate the token that would exceed the limit.
            if result.shape[1] >= max_ring_len:
                if debug_ring: print(f"[RANK {rank}] Iter {i} Stopping generation: Reached max ring length {max_ring_len}", flush=True)
                break
        # --- End Ring Attention Length Check ---

        input_ids = next_input[:, -max_seq_len:]

        # Debug 7: Confirm `input_ids` shape on each rank before each iteration
        if debug_ring:
            print(f"[RANK {rank}] Iter {i} input_ids shape for model call: {input_ids.shape}", flush=True)
        # Debug 8: Print current sequence (decoded) for each rank (assuming tokenizer available)
        if debug_ring and tokenizer:
            decoded = tokenizer.decode(result[0], skip_special_tokens=True) # Assuming batch size 1 for simplicity here
            print(f"[RANK {rank}] Iter {i} current decoded sequence: {decoded}", flush=True)

        # prepare any padding keyword arguments
        # iteration 0 is the prefill step (cache has not been filled yet), so no need to extend the mask/position_ids
        if i > 0:
            model_kwargs = __update_padding_kwargs(use_cache, model_kwargs)

        # position_ids may be generated by the hook, or we may need to
        # generate them here.
        if "position_ids" not in model_kwargs:
            # Determine the device of the input_ids, which should match the model's target device
            # (except potentially the embedding layer if it didn't move)
            expected_device = input_ids.device

            # if we are using kv cache, we need to handle position ids
            # incrementing from the length of the cache
            if use_cache and "past_key_value_states" in model_kwargs and model_kwargs["past_key_value_states"] is not None:
                # get the sequence length from the cache
                seq_len = model_kwargs["past_key_value_states"][0][0].shape[-2]
                # position is just the sequence length
                # Ensure position tensor is on the correct device initially
                position = torch.tensor([[seq_len]], dtype=torch.long, device=expected_device)
                model_kwargs["position_ids"] = position
            else:
                # otherwise, assume position is just the sequence length of inputs
                # Ensure position tensor is on the correct device initially
                position = torch.arange(input_ids.shape[1], dtype=torch.long, device=expected_device).unsqueeze(0)
                model_kwargs["position_ids"] = position

            # WORKAROUND for MPS/CPU mismatch with cached_freqs: Move position_ids to CPU
            if model_kwargs.get("position_ids") is not None:
                 model_kwargs["position_ids"] = model_kwargs["position_ids"].cpu()

        if prepare_model_inputs_hook is not None:
            input_ids, model_kwargs = prepare_model_inputs_hook(i, input_ids, model_kwargs)

        # Pass attn_algorithm and other model_kwargs to the model call
        output = model(input_ids, attn_algorithm=attn_algorithm, **model_kwargs)

        if use_cache:
            logits, past_key_value_states = output
            # TODO: this should go away when reduce-overhead issues are fixed, or
            # maybe could be moved into model code to be more portable.
            model_kwargs["past_key_value_states"] = past_key_value_states
            if contiguous_cache:
                model_kwargs["past_key_value_states"] = _make_cache_contiguous(
                    model_kwargs["past_key_value_states"]
                )
            if torch._dynamo.config.dynamic_shapes:
                model_kwargs["past_key_value_states"] = _make_cache_dynamic(
                    model_kwargs["past_key_value_states"]
                )
        else:
            logits = output

        # <<< WORKAROUND for Ring Attention returning global batch size >>>
        local_batch_size = result.shape[0] # The expected batch size for this rank
        if logits.shape[0] > local_batch_size and isinstance(strategy, RingAttentionStrategy):
            rank, _ = distributed.rank_and_world(strategy.group)
            start_idx = rank * local_batch_size
            end_idx = start_idx + local_batch_size
            if debug_ring: print(f"[RANK {rank}] Iter {i} Adjusting logits batch size from {logits.shape[0]} to {local_batch_size}", flush=True)
            logits = logits[start_idx:end_idx, :, :]
        # <<< END WORKAROUND >>>

        if "only_last_token" not in model_kwargs:
            logits = logits[:, -1, :]

        # Debug 17: Print if logits contain NaNs or infs
        if debug_ring:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"[RANK {rank}] Iter {i} logits contain NaNs or Infs!", flush=True)
        # Debug 18: Print logits mean/std
        if debug_ring:
            print(f"[RANK {rank}] Iter {i} logits stats: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}", flush=True)

        # --- Debug Ring Attention Logits (Sampling/Argmax Input) ---
        # Debug 4: Print logits top-5 indices and probabilities on each rank
        if debug_ring:
            # Always print top-5 from each rank if debugging
            top_probs, top_indices = torch.topk(F.softmax(logits.float(), dim=-1), 5)
            print(f"[RANK {rank}] Iter {i} Top-5 logits: Indices={top_indices.tolist()} :: Probs={top_probs.tolist()}", flush=True)

        # --- End Debug ---

        if do_sample:
            # get logits from last value in sequence nad scale
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            # --- Token Selection and Broadcast ---
            if is_distributed:
                # Allocate tensor to hold the next token on ALL ranks
                next_val = torch.empty((logits.size(0), 1), dtype=torch.long, device=device)

                if rank == 0:
                    # Calculate the next token only on rank 0 and copy to the tensor
                    calculated_next_val = torch.argmax(logits, dim=-1).unsqueeze(1) # Ensure shape [B, 1]
                    next_val.copy_(calculated_next_val)
                    # Debug 5: Print sampled/generated token on rank 0
                    if debug_ring:
                        print(f"[RANK 0] Iter {i} generated token: {next_val.tolist()}", flush=True)

                # Broadcast the chosen token from rank 0 to all other ranks
                dist.broadcast(next_val, src=0, group=strategy.group if isinstance(strategy, RingAttentionStrategy) else None)
                # Debug 9: Print confirmation after broadcast completes
                if debug_ring:
                    print(f"[RANK {rank}] Iter {i} broadcast completed", flush=True)
                # Debug 6: Print token received on non-zero ranks after broadcast
                if debug_ring and rank != 0:
                    print(f"[RANK {rank}] Iter {i} received token after broadcast: {next_val.tolist()}", flush=True)
                # Debug 10: Check all ranks have same token value after broadcast
                if debug_ring:
                    # Simple check by summing and dividing (assumes positive token IDs)
                    # Clone to avoid modifying the original tensor if needed elsewhere
                    token_check = next_val.clone().float()
                    # Use a temporary tensor for the sum to avoid modifying token_check in-place
                    token_sum = torch.zeros_like(token_check)
                    dist.all_reduce(token_check, op=dist.ReduceOp.SUM, group=strategy.group if isinstance(strategy, RingAttentionStrategy) else None)
                    token_sum.copy_(token_check) # Copy the result after all_reduce
                    expected_sum = next_val.item() * world_size # Assumes batch size 1 for simple check
                    print(f"[RANK {rank}] Iter {i} all-reduced token check (val={next_val.item()}, sum={token_sum.item()}, expected={expected_sum})", flush=True)
                    # assert token_sum.item() == expected_sum # Optional: hard assertion
            else: # Single process case
                next_val = torch.argmax(logits, dim=-1).unsqueeze(1) # Ensure shape [B, 1]


        if post_iteration_hook is not None:
            next_val, model_kwargs = post_iteration_hook(
                i + prompt_length, logits, next_val, model_kwargs
            )

        result = torch.cat((result, next_val), dim=-1)
        # Debug: Print result shape after concatenation on rank 0
        if debug_ring and rank == 0:
            print(f"[RANK 0] Iter {i} result shape after cat: {result.shape}", flush=True)

        # avoid continuing to generate if all have reached EOS
        if eos_token_id is not None:
            eos_found = torch.logical_or(eos_found, next_val == eos_token_id)
            if torch.sum(eos_found) == input_ids.shape[0]:
                if debug_ring: print(f"[RANK {rank}] Iter {i} Stopping generation: All sequences reached EOS.", flush=True)
                break

        # Check if we have reached the maximum sequence length (model's context window)
        if result.shape[1] >= max_seq_len:
            if debug_ring: print(f"[RANK {rank}] Iter {i} Stopping generation: Reached max sequence length {max_seq_len}", flush=True)
            break

        if use_cache:
            next_input = next_val
        else:
            next_input = result

        # Debug 15 cont'd: Print iteration time
        if debug_ring:
            print(f"[RANK {rank}] Iter {i} duration: {time.time() - iter_start_time:.3f}s", flush=True)
        # Debug 16: Print GPU memory usage per step (if available)
        if debug_ring and device.type == 'cuda': # Use the obtained device
            print(f"[RANK {rank}] Iter {i} CUDA memory used: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB", flush=True)

        if timing == "per-token":
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize()
            current_token_time = time.time() - start_time
            times.append(current_token_time)
            start_time = time.time()

    if timing == "e2e":
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize()
        e2e_time = time.time() - start_time
        times.append(e2e_time)

    if not is_batch:
        result = result[0]

    # Debug 19: At the end, print final generated sequence on rank 0 (assuming tokenizer available)
    if debug_ring and rank == 0 and tokenizer:
        final_decoded = tokenizer.decode(result, skip_special_tokens=True) # Use result directly if not is_batch handled it
        print(f"[RANK 0] Final raw result tokens: {result.tolist()}", flush=True) # Print raw tokens too
        print(f"[RANK 0] Final generated text: {final_decoded}", flush=True)
    # Debug 20: Confirm total number of tokens generated per rank
    if debug_ring:
        print(f"[RANK {rank}] Total tokens generated (incl. prompt): {result.shape[-1]}", flush=True)

    if timing != "":
        return result, times
    return result


def truncate_after_eos(
    result: torch.Tensor, eos_token_id: Union[int, Any, None]
) -> torch.Tensor:
    """
    Helper function to return a truncated sequence of token IDs stopping at
    (and including) the 'end of sentence' token.
    Currently only handles unbatched sequences.
    """
    if eos_token_id is None:
        return result
    eos_idx = torch.where(result == eos_token_id)
    eos_index = eos_idx[0]
    if eos_index.shape[0] >= 1:
        index = eos_index[0]
        result = result[: index + 1]
    return result


def trim_prefix(result: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Helper function to return a trimmed sequence of token IDs where
    all padding tokens (always 0 on our code) are removed.

    Examples:
    [0 0 0 0 1 2 3 4] with pad_token_id = 0 returns [1 2 3 4]
    [0 0 0 0 1 2 3 4] with pad_token_id = 5 returns [0 0 0 0 1 2 3 4]
    [1 2 3 4 0 1] with pad_token_id = 0 returns [1 2 3 4 0 1]

    Args:
    result: A 1D sequence of tokens
    pad_token_id: Token ID that will be trimmed from the start of the
        sequence
    """
    if result[0] != pad_token_id:
        return result
    output_diff = (result != pad_token_id).diff()
    first_real_token_idx = torch.where(output_diff > 0)
    if first_real_token_idx[0].numel() == 0:
        return result
    bos_index = first_real_token_idx[0][0]
    result = result[bos_index + 1 :]
    return result
