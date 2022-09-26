# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
# 2022.07 - Modified the code to support Mindspore
#           Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Decoder definition."""

from typing import Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

from mindaudio.models.conformer.layers.dense import Dense
from mindaudio.models.conformer.layers.layernorm import LayerNorm
from mindaudio.models.conformer.transformer.attention import MultiHeadedAttention
from mindaudio.models.conformer.transformer.decoder_layer import DecoderLayer
from mindaudio.models.conformer.transformer.embedding import PositionalEncoding
from mindaudio.models.conformer.transformer.positionwise_feed_forward import PositionwiseFeedForward
from mindaudio.utils.common import get_activation


class TransformerDecoder(nn.Cell):
    """Base class of Transformer decoder module.

    Args:
        vocab_size (int): output dim.
        encoder_output_size (int): dimension of attention.
        attention_heads (int): the number of heads of multi head attention.
        linear_units (int): the hidden units number of position-wise feedforward.
        num_blocks (int): the number of decoder blocks.
        dropout_rate (float): dropout rate.
        positional_dropout_rate (float): dropout rate positional encoding.
        self_attention_dropout_rate (float): dropout rate for self-attention.
        src_attention_dropout_rate (float): dropout rate for src-attention.
        input_layer (str): input layer type.
        use_output_layer (bool): whether to use output layer.
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after (bool): whether to concat attention layer's input and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.0,
                 src_attention_dropout_rate: float = 0.0,
                 input_layer: str = 'embed',
                 use_output_layer: bool = True,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 compute_type=mstype.float32):
        super().__init__()
        attention_dim = encoder_output_size
        activation = get_activation('relu')
        self.first_flag = True

        if input_layer == 'embed':
            self.embed = nn.SequentialCell(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.use_output_layer = use_output_layer

        if use_output_layer:
            self.output_layer = Dense(attention_dim, vocab_size).to_float(compute_type)
        if normalize_before:
            self.after_norm = LayerNorm(attention_dim, epsilon=1e-12)

        self.decoders = nn.CellList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    compute_type,
                ),
                MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    src_attention_dropout_rate,
                    compute_type,
                ),
                PositionwiseFeedForward(
                    attention_dim,
                    linear_units,
                    dropout_rate,
                    activation,
                    compute_type,
                ),
                dropout_rate,
                normalize_before,
                concat_after,
                compute_type,
            ) for _ in range(num_blocks)
        ])
        self.expand_dims = ops.ExpandDims()
        self.log_softmax = nn.LogSoftmax()

    def construct(self, memory: mindspore.Tensor, memory_mask: mindspore.Tensor, ys_in_pad: mindspore.Tensor,
                  ys_masks: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_masks: mask for token sequences
        Returns:
            x: decoded token score before softmax (batch, maxlen_out,
                vocab_size) if use_output_layer is True
        """
        x, _ = self.embed(ys_in_pad)
        for layer in self.decoders:
            x, ys_masks, memory, memory_mask = layer(x, ys_masks, memory, memory_mask)

        if self.normalize_before:
            x = self.after_norm(x)

        if self.use_output_layer:
            x = self.output_layer(x)

        return x
