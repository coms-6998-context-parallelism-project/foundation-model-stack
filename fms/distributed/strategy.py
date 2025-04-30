import os
from abc import abstractmethod
from typing import List, Tuple

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from fms.utils import tp_wrapping
from fms import distributed


if "DISTRIBUTED_STRATEGY_IGNORE_MODULES" in os.environ:
    _distributed_strategy_ignore_modules = os.environ[
        "DISTRIBUTED_STRATEGY_IGNORE_MODULES"
    ].split(",")
else:
    _distributed_strategy_ignore_modules = []


class DistributedStrategy:
    def __init__(self, from_meta=False):
        self.from_meta = from_meta

    def __should_distribute(self, module_name: str) -> bool:
        return module_name not in _distributed_strategy_ignore_modules

    def distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        """
        Optionally a distributed strategy may distribute modules that are not
        numbered layers
        """
        module_name = type(module).__name__
        if self.__should_distribute(module_name):
            return self._distribute_module(module, final_layers)
        else:
            print(f"ignoring module={module_name} when distributing module")
            return module

    def distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        """
        Distribute each layer as-appropriate
        """
        block_name = type(block).__name__
        if self.__should_distribute(block_name):
            return self._distribute_layer(block, layer)
        else:
            print(f"ignoring block={block_name} when distributing layer")
            return block

    @abstractmethod
    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        """
        Distribute modules that are not numbered layers
        """
        pass

    @abstractmethod
    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        """
        Distribute each layer
        """
        pass


class NotDistributed(DistributedStrategy):
    def __init__(self, from_meta=False):
        super().__init__(from_meta)

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return block


NoOpStrategy = NotDistributed()


class DeviceMover(nn.Module):
    def __init__(self, module: nn.Module, device):
        super().__init__()
        self.device = device
        # make this wrapper module behave as if it was the wrapped module.
        attr = module.__dict__
        attr["module"] = module.to(device)
        attr["device"] = device
        self.__dict__ = attr

    def forward(self, *args, **kwargs):
        device = self.device
        args = [
            arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        kwargs = {
            k: (
                kwargs[k].to(device)
                if isinstance(kwargs[k], torch.Tensor)
                else kwargs[k]
            )
            for k in kwargs
        }
        return self.module(*args, **kwargs)


class UniformModelParallelStrategy(DistributedStrategy):
    def __init__(self, devices: List[int], num_layers: int, from_meta=False):
        super().__init__(from_meta)
        num_dev = len(devices)
        layers_per_dev = num_layers // num_dev
        remainder = num_layers - (layers_per_dev * num_dev)
        self.layer_to_device = [0] * num_layers
        layer_id = 0
        for dev_idx in range(len(devices)):
            for i in range(layers_per_dev):
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
            if remainder > 0:
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
                remainder -= 1

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        device = self.layer_to_device[layer]
        if self.from_meta:
            # https://github.com/pytorch/pytorch/pull/113647
            block.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(block, device)
        return wrapped

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        if final_layers:
            device = self.layer_to_device[len(self.layer_to_device) - 1]
        else:
            device = self.layer_to_device[0]
        if self.from_meta:
            return module.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(module, device)
        return wrapped


class TensorParallelStrategy(DistributedStrategy):
    def __init__(self, group=None, from_meta=False):
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "must initialize a process group"
        self.group = group if group is not None else torch.distributed.GroupMember.WORLD

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        return tp_wrapping.apply_tp(module, self.group)

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return tp_wrapping.apply_tp(block, self.group)


import math
from typing import Tuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

# ... (keep existing imports like abc, os, List, tp_wrapping, distributed)

# ... (keep existing DistributedStrategy, NotDistributed, NoOpStrategy, DeviceMover, UniformModelParallelStrategy, TensorParallelStrategy classes)


class RingAttentionStrategy(DistributedStrategy):
    """
    A strategy designed for Ring Attention where the sequence dimension is sharded
    across devices in a group using a fixed block size. Input is globally padded,
    sharded into fixed blocks, computation happens on shards, and output is gathered.
    """
    def __init__(self, block_size: int, group=None, from_meta=False):
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "must initialize a process group"
        self.group = group if group is not None else torch.distributed.GroupMember.WORLD
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.block_size = block_size
        self.dim = 1 # Assume sequence dimension is 1 for [batch, seq, hidden]

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        # Ring Attention typically doesn't require specific module sharding like TP.
        # Modules are replicated, and data is sharded.
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        # Layers run on their local rank with sharded data. No specific layer distribution needed here.
        return block

    def shard_input(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Validates sequence length, pads globally to world_size * block_size,
        and shards the input tensor along the sequence dimension into fixed blocks.
        Returns the local shard and the original global sequence length.
        """
        if self.world_size == 1:
            # No sharding or padding needed for single GPU case
            return tensor, tensor.size(self.dim)

        original_global_seq_len = tensor.size(self.dim)
        global_shape = tensor.shape

        # 1. Validation: Check if the sequence fits within world_size * block_size
        required_gpus = math.ceil(original_global_seq_len / self.block_size)
        if required_gpus > self.world_size:
            raise ValueError(
                f"Sequence length {original_global_seq_len} requires {required_gpus} blocks/GPUs "
                f"with block_size {self.block_size}, but only {self.world_size} GPUs are available."
            )

        # 2. Global Padding: Pad the input tensor so its total length is world_size * block_size
        padded_global_len = self.world_size * self.block_size
        global_pad_len = padded_global_len - original_global_seq_len

        if global_pad_len < 0:
             # This should ideally not happen if validation passes, but as a safeguard.
             raise ValueError(f"Negative padding calculated ({global_pad_len}). Padded length {padded_global_len} < Original length {original_global_seq_len}")

        if global_pad_len > 0:
            # Calculate padding tuple based on self.dim
            # Example: dim=1 in 3D tensor [B, S, H] -> pad=(0, 0, 0, global_pad_len)
            padding_dims = [0] * (tensor.dim() * 2)
            # Index for padding at the end of the target dimension (self.dim)
            # F.pad pads from the last dim backwards, so index is -(2*dim_index_from_end + 1)
            dim_index_from_end = tensor.dim() - 1 - self.dim
            padding_dims[2 * dim_index_from_end + 1] = global_pad_len
            # Use 0 for padding value, assuming it's appropriate (e.g., for input_ids if 0 is pad_token_id)
            tensor_padded = F.pad(tensor, tuple(padding_dims), "constant", 0) # Pad with 0
            # print(f"[rank{self.rank}] RingAttentionStrategy.shard_input: Globally padded input from seq len {original_global_seq_len} to {padded_global_len}", flush=True)
        else:
            tensor_padded = tensor
            # print(f"[rank{self.rank}] RingAttentionStrategy.shard_input: No global padding needed for seq len {original_global_seq_len}", flush=True)

        # 3. Sharding: Each rank gets exactly one block of size block_size
        start_idx = self.rank * self.block_size
        # Slice the globally padded tensor
        local_shard = tensor_padded.narrow(self.dim, start_idx, self.block_size).contiguous() # Each shard is block_size

        print(f"[rank{self.rank}] RingAttentionStrategy.shard_input: Original global shape {global_shape}, Padded global shape {tensor_padded.shape}, Local shard shape {local_shard.shape} (dim={self.dim})", flush=True)
        # Return the shard (always size block_size) and the original global length for trimming later
        return local_shard, original_global_seq_len

    def gather_output(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gathers tensors (expected to be fixed block_size) from all ranks along the sequence dimension."""
        if self.world_size == 1:
            return tensor

        # print(f"[rank{self.rank}] RingAttentionStrategy.gather_output: Local shape before gather {tensor.shape} (dim={self.dim})", flush=True)

        # Ensure the local tensor has the expected block size dimension
        if tensor.shape[self.dim] != self.block_size:
             # This might happen if the model's forward pass changes the sequence length unexpectedly
             raise RuntimeError(f"[rank{self.rank}] Expected local output shard size {self.block_size} along dim {self.dim}, but got {tensor.shape[self.dim]}.")

        # Determine the shape of the fully gathered tensor (padded) on each GPU
        global_shape = list(tensor.shape)
        # The gathered dimension will be world_size * block_size
        global_shape[self.dim] = self.world_size * self.block_size

        # Store original device
        original_device = tensor.device

        # --- Use dist.all_gather instead of all_gather_into_tensor ---
        # Create a list to hold the gathered tensors from all ranks
        # Ensure the list tensors are on CPU for Gloo backend
        output_list = [torch.empty_like(tensor, device='cpu') for _ in range(self.world_size)]

        # Perform all-gather directly into the tensor
        # Ensure local_output is contiguous if required by the backend
        # Move tensor to CPU before gathering for Gloo backend
        dist.all_gather(output_list, tensor.contiguous().cpu(), group=self.group)

        # Concatenate the gathered tensors along the specified dimension
        gathered_output_cpu = torch.cat(output_list, dim=self.dim)
        # --- End change ---

        # Move the final result back to the original device
        gathered_output = gathered_output_cpu.to(original_device)
        # print(f"[rank{self.rank}] RingAttentionStrategy.gather_output: Global shape after gather {gathered_output.shape} (device: {gathered_output.device})", flush=True)
        # Return the gathered tensor, which includes the global padding
        return gathered_output

# at all