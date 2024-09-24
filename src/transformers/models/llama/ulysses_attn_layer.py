# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Any
from torch import Tensor
import torch.distributed as dist
from flash_attn import flash_attn_func
from .all_to_all import SeqAllToAll4D, SeqAllToAllAsync4D
import torch.nn.functional as F
from transformers import parallel_state as mpu
from .stream_pool import CudaStreamPool


class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_fa : bool = True 
    ) -> None:

        super(UlyssesAttention, self).__init__()
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_fa = use_fa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            self.use_fa = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        fa_fn,
        *args: Any
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        seq_parallel_group = mpu.get_sequence_ulysses_parallel_group()

        q = SeqAllToAll4D.apply(seq_parallel_group, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(seq_parallel_group, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(seq_parallel_group, value, self.scatter_idx, self.gather_idx)

        context_layer = fa_fn(q, k, v)

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            seq_parallel_group, context_layer, self.gather_idx, self.scatter_idx
        )
        # print(f"{output=}")
        # out e.g., [s/p::h]
        return output

class UlyssesAsyncAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_fa : bool = True
    ) -> None:

        super(UlyssesAsyncAttention, self).__init__()
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_fa = use_fa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        self.stream = None
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            self.use_fa = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        fa_fn,
        *args: Any
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        seq_parallel_group = mpu.get_sequence_ulysses_parallel_group()

        q, k, v = SeqAllToAllAsync4D.apply(seq_parallel_group, query, key, value, self.scatter_idx, self.gather_idx, self.stream)

        context_layer = fa_fn(q, k, v)
        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            seq_parallel_group, context_layer, self.gather_idx, self.scatter_idx
        )
        # out e.g., [s/p::h]
        return output
