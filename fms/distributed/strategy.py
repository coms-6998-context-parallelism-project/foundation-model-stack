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


class RingAttentionStrategy(DistributedStrategy):
    """
    A strategy designed for Ring Attention where the sequence dimension is sharded
    across devices in a group. Input is sharded, computation happens on shards,
    and output is gathered.
    Assumes simple block sharding for sequence dimension.
    """
    def __init__(self, group=None, from_meta=False):
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "must initialize a process group"
        self.group = group if group is not None else torch.distributed.GroupMember.WORLD
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        # Ring Attention typically doesn't require specific module sharding like TP.
        # Modules are replicated, and data is sharded.
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        # Layers run on their local rank with sharded data. No specific layer distribution needed here.
        return block

    def get_local_seq_len_and_offset(self, global_seq_len: int) -> Tuple[int, int]:
        """Calculates the length and starting offset for the local shard."""
        shard_len = (global_seq_len + self.world_size - 1) // self.world_size
        start = self.rank * shard_len
        end = min((self.rank + 1) * shard_len, global_seq_len)
        local_len = max(0, end - start)
        return local_len, start

    def shard_input(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Shards the input tensor along a given dimension."""
        if self.world_size == 1:
            return tensor
        global_len = tensor.size(dim)
        global_shape = tensor.shape
        local_len, start_offset = self.get_local_seq_len_and_offset(global_len)
        local_shard = tensor.narrow(dim, start_offset, local_len).contiguous()
        print(f"[rank{self.rank}] RingAttentionStrategy.shard_input: Global shape {global_shape}, Local shard shape {local_shard.shape} (dim={dim})")
        return local_shard

    def gather_output(self, tensor: torch.Tensor, dim: int, global_len: int) -> torch.Tensor:
        """Gathers tensors from all ranks along a given dimension."""
        if self.world_size == 1:
            return tensor
        print(f"[rank{self.rank}] RingAttentionStrategy.gather_output: Local shape before gather {tensor.shape} (dim={dim})")
        # Use all_gather_into_tensor for potentially better performance, requires knowing output size
        # Output tensor needs to accommodate the full global length
        output_shape = list(tensor.shape)
        output_shape[dim] = global_len # Use the provided global_len which should be padded
        output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

        # Pad the local tensor if it's smaller than the largest shard size before gathering
        # all_gather_into_tensor requires all input tensors to have the same size.
        max_shard_len = (global_len + self.world_size - 1) // self.world_size # Size of the largest shard
        local_len = tensor.shape[dim]
        padding_needed = max_shard_len - local_len

        if padding_needed > 0:
            pad_dims = [0] * (tensor.dim() * 2)
            # Target the specific dimension for padding (F.pad works from last dim backwards)
            pad_idx = tensor.dim() - 1 - dim
            pad_dims[2 * pad_idx + 1] = padding_needed # Pad at the end of the target dimension
            padded_tensor = F.pad(tensor, pad_dims, value=0) # Pad with 0
            tensor_to_gather = padded_tensor
        else:
            tensor_to_gather = tensor

        dist.all_gather_into_tensor(output_tensor, tensor_to_gather, group=self.group)
        print(f"[rank{self.rank}] RingAttentionStrategy.gather_output: Global shape after gather {output_tensor.shape}")
        return output_tensor
