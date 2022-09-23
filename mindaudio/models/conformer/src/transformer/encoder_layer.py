# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

from mindaudio.models.conformer.src.layers.dense import Dense
from mindaudio.models.conformer.src.layers.layernorm import LayerNorm


class TransformerEncoderLayer(nn.Cell):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 size: int,
                 self_attn: nn.Cell,
                 feed_forward: nn.Cell,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 compute_type=mstype.float32):
        """Construct an EncoderLayer object."""
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size, epsilon=1e-5)
        self.norm2 = LayerNorm(size, epsilon=1e-5)
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Dense(size + size, size).to_float(compute_type)
        self.concat_dimlast = ops.Concat(-1)
        self.concat_dim1 = ops.Concat(1)

    def construct(
            self,
            x: mindspore.Tensor,
            mask: mindspore.Tensor,
            output_cache: Optional[mindspore.Tensor] = None
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Compute encoded features.

        Args:
            x (minspore.Tensor): Input tensor (#batch, time, size).
            mask (minspore.Tensor): Mask tensor for the input (#batch, 1, time).
            output_cache (minspore.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
        Returns:
            minspore.Tensor: Output tensor (#batch, time, size).
            minspore.Tensor: Mask tensor (#batch, time).
        """
        # Multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        if self.concat_after:
            x_concat = self.concat_dimlast((x, self.self_attn(x_q, x, x, mask)))
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        # Feedforawrd module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = self.concat_dim1([output_cache, x], dim=1)

        return x, mask


class ConformerEncoderLayer(nn.Cell):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 size: int,
                 self_attn: nn.Cell,
                 feed_forward: nn.Cell,
                 feed_forward_macaron: nn.Cell,
                 conv_module: nn.Cell,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 compute_type=mstype.float32):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.ff_scale = 0.5
        self.norm_ff = LayerNorm(size, epsilon=1e-5)
        self.norm_mha = LayerNorm(size, epsilon=1e-5)
        self.norm_ff_macaron = LayerNorm(size, epsilon=1e-5)
        self.norm_conv = LayerNorm(size, epsilon=1e-5)
        self.norm_final = LayerNorm(size, epsilon=1e-5)
        self.dropout = nn.Dropout(1 - dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Dense(size + size, size).to_float(compute_type)
        self.concat_dim1 = ops.Concat(1)
        self.concat_dimlast = ops.Concat(-1)
        self.cast = ops.Cast()
        self.get_dtype = ops.DType()
        self.compute_type = compute_type

    def construct(
            self,
            x: mindspore.Tensor,
            mask: mindspore.Tensor,
            pos_emb: mindspore.Tensor,
            mask_pad: mindspore.Tensor,
            output_cache: Optional[mindspore.Tensor] = None
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Compute encoded features.

        Args:
            x (minspore.Tensor): (#batch, time, size)
            mask (minspore.Tensor): Mask tensor for the input (#batch, 1, time).
            pos_emb (minspore.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (mindspore.Tensor): mask for input tensor.
            output_cache (minspore.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
        Returns:
            minspore.Tensor: Output tensor (#batch, time, size).
            minspore.Tensor: Mask tensor (#batch, time).
        """
        # Macaron-Net Feedforward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff_macaron(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
        if not self.normalize_before:
            x = self.norm_ff_macaron(x)

        # Multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if output_cache is None:
            x_q = x
        else:
            # TODO: wait to be reviewed
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, mask, pos_emb)

        # TODO: need to be reviewed
        if self.concat_after:
            x_concat = self.concat_dimlast((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution module
        residual = x
        if self.normalize_before:
            x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x, mask_pad))
        if not self.normalize_before:
            x = self.norm_conv(x)

        # Feedforward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        # Final normalization
        x = self.norm_final(x)

        if output_cache is not None:
            x = self.concat_dim1([output_cache, x], dim=1)

        return x, mask
