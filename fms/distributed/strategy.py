import os
from abc import abstractmethod
from typing import List, Optional

import torch
import torch.distributed
from torch import nn

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


from torch import Tensor
import torch.distributed as dist

class RingAttentionStrategy(DistributedStrategy):
    """Distributed strategy for ring attention with fixed block size."""

    def __init__(self, block_size: int = 16, group: Optional[dist.ProcessGroup] = None, from_meta: bool = False):
        super().__init__(from_meta)
        self.block_size = block_size
        
        if dist.is_initialized():
            self.group = group if group is not None else dist.GroupMember.WORLD
            self.rank: int = self.group.rank()
            self.world_size: int = self.group.size()
        else: # Handle non-distributed case (e.g. nproc=1 local run)
            self.group = None # type: ignore
            self.rank = 0
            self.world_size = 1
            # print0("[INFO] RingAttentionStrategy: torch.distributed not initialized. Defaulting to world_size=1, rank=0.")

        # State tracked per forward pass
        self._original_seq_len: Optional[int] = None
        self._local_valid_len: Optional[int] = None # Tokens on this rank before padding

    # No-op for distributing modules/layers in Ring Attention
    def _distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module: return module
    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module: return block

    def _pad_to_block_size(self, x: Tensor, dim: int = 1) -> Tensor:
        """Pads tensor along dim to self.block_size."""
        length = x.size(dim)
        if length == self.block_size: return x
        assert length < self.block_size, f"Rank {self.rank}: Len {length} >= block_size {self.block_size} on dim {dim}"
        pad_shape = list(x.shape)
        pad_shape[dim] = self.block_size - length
        pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=dim)

    def shard_input(self, x: Tensor) -> Tensor:
        """Shards input along dim 1, pads shard to block_size."""
        if self.world_size == 1:
            self._original_seq_len = x.size(1)
            self._local_valid_len = x.size(1)
            return x

        seq_len = x.size(1)
        self._original_seq_len = seq_len

        start = self.rank * self.block_size
        end = min(start + self.block_size, seq_len)
        self._local_valid_len = max(0, end - start) # Valid tokens on this rank

        shard = x[:, start:end, ...] if self._local_valid_len > 0 else torch.empty_like(x[:, :0, ...])
        padded_shard = self._pad_to_block_size(shard, dim=1)

        # Rigor: Assert output shape
        assert padded_shard.size(1) == self.block_size, f"Rank {self.rank}: Shard pad failed: {padded_shard.size(1)} != {self.block_size}"
        assert padded_shard.device == x.device, f"Rank {self.rank}: Shard device mismatch"
        assert padded_shard.dtype == x.dtype, f"Rank {self.rank}: Shard dtype mismatch"

        return padded_shard

    def shard_global_position_ids(self, global_position_ids: Tensor, target_rank: int) -> Tensor:
        """
        Takes global position_ids and returns the specific shard for target_rank, padded to block_size.
        Assumes global_position_ids are (B, Global_Seq_Len).
        """
        if self.world_size == 1:
            # If only one rank, it gets the whole (potentially already block-sized) tensor.
            # No sharding needed, but ensure it's padded if it's shorter than block_size.
            return self._pad_to_block_size(global_position_ids, dim=1)

        B, global_seq_len = global_position_ids.shape
        
        start = target_rank * self.block_size
        end = min(start + self.block_size, global_seq_len)
        local_valid_len_for_target_rank = max(0, end - start)

        shard = global_position_ids[:, start:end] if local_valid_len_for_target_rank > 0 else torch.empty_like(global_position_ids[:, :0])
        padded_shard = self._pad_to_block_size(shard, dim=1)
        assert padded_shard.size(1) == self.block_size, f"Rank {self.rank} (sharding for {target_rank}): Pos ID shard pad failed: {padded_shard.size(1)} != {self.block_size}"
        return padded_shard

    def get_local_valid_len(self) -> int:
        """Returns the valid (non-padded) sequence length on the local rank."""
        assert self._local_valid_len is not None, "get_local_valid_len called before shard_input."
        return self._local_valid_len

    def gather_tensor(self, tensor: Tensor, dim: int = 1) -> Tensor:
        """Gathers tensors from all ranks along dim, pads, concatenates, slices if dim=1."""
        if self.world_size == 1: return tensor

        # Pad tensor on this rank before gathering
        tensor_padded = self._pad_to_block_size(tensor, dim=dim)
        assert tensor_padded.size(dim) == self.block_size, f"Rank {self.rank}: Gather input pad failed: {tensor_padded.size(dim)} != {self.block_size} on dim {dim}"
        assert tensor_padded.device == tensor.device, f"Rank {self.rank}: Gather device mismatch"
        assert tensor_padded.dtype == tensor.dtype, f"Rank {self.rank}: Gather dtype mismatch"


        gathered_list = [torch.empty_like(tensor_padded) for _ in range(self.world_size)]
        dist.all_gather(gathered_list, tensor_padded.contiguous(), group=self.group)
        result = torch.cat(gathered_list, dim=dim)

        # If gathering seq dim, slice back to original length
        if dim == 1:
            assert self._original_seq_len is not None, f"Rank {self.rank}: gather_tensor (dim=1) called before shard_input."
            expected_cat_len = self.block_size * self.world_size
            assert result.size(dim) == expected_cat_len, f"Rank {self.rank}: Gather concat failed: {result.size(dim)} != {expected_cat_len}"
            result = result.narrow(dim, 0, self._original_seq_len)
            assert result.size(dim) == self._original_seq_len, f"Rank {self.rank}: Gather slice failed: {result.size(dim)} != {self._original_seq_len}"

        return result

    def gather_output(self, tensor_to_gather: Tensor, *args, **kwargs) -> Tensor:
        """
        Gathers tensors from all ranks. This is a new method based on the modified calls.
        For now, it can delegate to gather_tensor.
        """
        return self.gather_tensor(tensor_to_gather, dim=1) # Assuming dim 1 is appropriate
