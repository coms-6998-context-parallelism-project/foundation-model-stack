import os
from abc import abstractmethod
import math # Add math import
from typing import List

import torch
import torch.distributed
from torch import nn, Tensor, distributed as dist

from fms.utils import tp_wrapping


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


class RingAttentionStrategy(DistributedStrategy):
    """
    Distributed strategy for ring attention with automatic input padding.
    Handles input sharding and output gathering across ranks.
    """

    def __init__(self, block_size: int, group=None, from_meta=False): # Add block_size
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "Requires initialized process group"
        self.group = group or torch.distributed.GroupMember.WORLD
        self.rank = self.group.rank()
        self.world_size = self.group.size()
        self.block_size = block_size # Store block_size
        self._original_seq_len = None  # Track original sequence length for unpadding

    def _distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module:
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return block

    def shard_input(self, x: Tensor) -> Tensor:
        """
        Shard input along the sequence dimension. Automatically pads if needed.
        """
        # If world_size is 1, no sharding needed, but store original length
        if self.world_size == 1:
            self._original_seq_len = x.size(1)
            return x

        batch_size, seq_len = x.size(0), x.size(1)
        self._original_seq_len = seq_len

        # Calculate start and end indices for this rank's block
        start_idx = self.rank * self.block_size

        # Handle ranks that are completely outside the original sequence length
        if start_idx >= seq_len:
            # Return a tensor of size block_size, filled with padding (zeros)
            # This ensures all ranks have a tensor of the same size for gather operations
            return torch.zeros((batch_size, self.block_size, *x.shape[2:]), dtype=x.dtype, device=x.device)

        end_idx = min(start_idx + self.block_size, seq_len)
        shard = x[:, start_idx:end_idx, :].contiguous()

        # Pad the shard if it's smaller than block_size (occurs for the last block)
        current_shard_len = shard.size(1)
        pad_len = self.block_size - current_shard_len
        if pad_len > 0:
            pad_tensor = torch.zeros((batch_size, pad_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
            shard = torch.cat([shard, pad_tensor], dim=1)

        return shard

    def gather_output(self, x_local: Tensor) -> Tensor:
        """
        Gather sequence shards and trim padding to recover original sequence length.
        """
        if self.world_size == 1:
            return x_local

        # x_local is the output corresponding to the (potentially padded) shard.
        # It should have sequence length self.block_size.
        gathered = [torch.empty_like(x_local) for _ in range(self.world_size)]
        dist.all_gather(gathered, x_local.contiguous(), group=self.group)
        full_padded = torch.cat(gathered, dim=1) # Shape: [B, world_size * block_size, D]

        # Trim the result back to the original sequence length.
        # The gathered tensor might be longer than original_seq_len due to padding of shards.
        if self._original_seq_len is not None and full_padded.size(1) > self._original_seq_len:
            full = full_padded[:, :self._original_seq_len] # Slice full_padded and assign to full
        else:
            full = full_padded # If no trimming needed, assign full_padded to full
        return full

    def gather_tensor(self, tensor: Tensor, dim: int = 1) -> Tensor:
        """
        Gathers a tensor sharded along a specific dimension across ranks.
        Assumes the tensor might have been padded by shard_input if dim=1.

        Args:
            tensor (Tensor): The local shard of the tensor.
            dim (int): The dimension along which the tensor is sharded. Defaults to 1 (sequence dim).

        Returns:
            Tensor: The gathered, potentially unpadded, tensor on all ranks.
        """
        if self.world_size == 1:
            return tensor

        gathered_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_list, tensor.contiguous(), group=self.group)
        gathered_tensor = torch.cat(gathered_list, dim=dim)

        # Trim padding if gathering along sequence dimension (dim=1)
        # and the original sequence length was stored (meaning shard_input was called)
        # and padding might have been applied to shards.
        # Assumes the gathered tensor's dimension `dim` corresponds to the sequence length.
        if dim == 1 and self._original_seq_len is not None and gathered_tensor.size(dim) > self._original_seq_len:
             gathered_tensor = gathered_tensor.narrow(dim, 0, self._original_seq_len)
        return gathered_tensor
