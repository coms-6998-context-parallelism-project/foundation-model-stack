import logging
import math
import queue
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class LLaMAConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    emb_dim: int = 4096
    
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 0
    nlayers: int = 32
    pad_id: int = -1
    hidden_grow_factor: float = 8 / 3
    multiple_of: int = 256
    activation_fn: str = "swish"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 4096
    ntk_scaling: bool = False
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = False
    rope_theta: float = 10_000.0
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True



@dataclass
class BlockData:
    engine_instance: 'LLaMABlock' 
    block_id: int
    num_blocks: int
    q_start: int
    q_end: int
    mask_global: Optional[Tensor]
    block_queues: List[queue.Queue]
    await_max: threading.Barrier
    await_sums: threading.Barrier
    result_buffer: Dict[int, Tensor]
    q_block: Tensor
    k_local: Tensor
    v_local: Tensor
    x_block: Tensor


