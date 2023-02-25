# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
# 2022.07 - Modified the code to support Mindspore
#           Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decoder self-attention layer definition."""

from typing import Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

from flyspeech.layers.dense import Dense
from flyspeech.layers.layernorm import LayerNorm


class DecoderLayer(nn.Cell):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (nn.Cell): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (nn.Cell): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (nn.Cell): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 size: int,
                 self_attn: nn.Cell,
                 src_attn: nn.Cell,
                 feed_forward: nn.Cell,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 compute_type=mstype.float32):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size, epsilon=1e-12)
        self.norm2 = LayerNorm(size, epsilon=1e-12)
        self.norm3 = LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = Dense(size + size, size).to_float(compute_type)
            self.concat_linear2 = Dense(size + size, size).to_float(compute_type)
        self.cat1 = ops.Concat(axis=-1)
        self.cat2 = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.cast = ops.Cast()

    def construct(
            self, tgt: mindspore.Tensor, tgt_mask: mindspore.Tensor, memory: mindspore.Tensor,
            memory_mask: mindspore.Tensor
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Compute decoded features.

        Args:
            tgt (mindspore.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (mindspore.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (mindspore.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (mindspore.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (mindspore.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            mindspore.Tensor: Output tensor (#batch, maxlen_out, size).
            mindspore.Tensor: Mask for output tensor (#batch, maxlen_out).
            mindspore.Tensor: Encoded memory (#batch, maxlen_in, size).
            mindspore.Tensor: Encoded memory mask (#batch, maxlen_in).
        """
        # Self-attention module
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        tgt_q = tgt
        tgt_q_mask = tgt_mask
        if self.concat_after:
            tgt_concat = self.cat1((tgt_q, self.cast(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask), tgt_q.dtype)))
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # Src-attention module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        if self.concat_after:
            x_concat = self.cat1((x, self.cast(self.src_attn(x, memory, memory, memory_mask), x.dtype)))
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        # Feedforward module
        residual = x
        if self.normalize_before:
            x = self.norm3(x)

        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, tgt_mask, memory, memory_mask
