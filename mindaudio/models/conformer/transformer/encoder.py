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
"""Encoder definition."""

from typing import Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn

from mindaudio.models.conformer.layers.layernorm import LayerNorm
from mindaudio.models.conformer.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from mindaudio.models.conformer.transformer.convolution import ConvolutionModule
from mindaudio.models.conformer.transformer.embedding import (
    ConvPositionalEncoding, 
    NoPositionalEncoding, 
    PositionalEncoding,
    RelPositionalEncoding
)
from mindaudio.models.conformer.transformer.encoder_layer import ConformerEncoderLayer, TransformerEncoderLayer
from mindaudio.models.conformer.transformer.positionwise_feed_forward import PositionwiseFeedForward
from mindaudio.models.conformer.transformer.subsampling import Conv2dSubsampling4
from mindaudio.utils.common import get_activation


class BaseEncoder(nn.Cell):
    """Base encode instance.

    Args:
        input_size (int): input dim
        output_size (int): dimension of attention
        positional_dropout_rate (float): dropout rate after adding
            positional encoding
        input_layer (str): input layer type.
            optional [linear, conv2d, conv2d6, conv2d8]
        pos_enc_layer_type (str): Encoder positional encoding layer type.
            opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        feature_norm (bool): whether do feature norm to input features, like CMVN
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 positional_dropout_rate: float = 0.1,
                 input_layer: str = 'conv2d',
                 pos_enc_layer_type: str = 'abs_pos',
                 normalize_before: bool = True,
                 feature_norm: bool = True,
                 global_cmvn: mindspore.nn.Cell = None,
                 compute_type=mindspore.float32):
        """construct BaseEncoder."""
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == 'abs_pos':
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == 'rel_pos':
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == 'conv_pos':
            pos_enc_class = ConvPositionalEncoding
        else:
            pos_enc_class = NoPositionalEncoding
        self.input_layer = input_layer
        if input_layer == 'conv2d':
            subsampling_class = Conv2dSubsampling4
            self.embed = subsampling_class(input_size, output_size, pos_enc_class(output_size, positional_dropout_rate),
                                           compute_type)
        else:
            self.embed = pos_enc_class(output_size, positional_dropout_rate)

        self.normalize_before = normalize_before
        if normalize_before:
            self.after_norm = LayerNorm(output_size, epsilon=1e-5)

        self.feature_norm = feature_norm
        self.global_cmvn = global_cmvn

    def output_size(self) -> int:
        return self._output_size

    def construct(self,
                  xs: mindspore.Tensor,
                  masks: mindspore.Tensor,
                  xs_chunk_masks: mindspore.Tensor = None) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            masks: masks for the input xs ()
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: mindspore.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        if self.global_cmvn:
            xs = self.global_cmvn(xs)

        # masks is subsampled to (B, 1, T/subsample_rate)
        # print('[encoder] embed:', self.embed)
        # print('[encoder] xs:', xs.shape)
        xs, pos_emb = self.embed(xs)
        for layer in self.encoders:
            xs, xs_chunk_masks = layer(xs, xs_chunk_masks, pos_emb, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module.

    Args:
        input_size (int): input dim
        output_size (int): dimension of attention
        attention_heads (int): the number of heads of multi head attention
        linear_units (int): the hidden units number of position-wise feed
            forward
        num_blocks (int): the number of decoder blocks
        dropout_rate (float): dropout rate
        attention_dropout_rate (float): dropout rate in attention
        positional_dropout_rate (float): dropout rate after adding
            positional encoding
        input_layer (str): input layer type.
            optional [linear, conv2d, conv2d6, conv2d8]
        pos_enc_layer_type (str): Encoder positional encoding layer type.
            opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        feature_norm (bool): whether do feature norm to input features, like CMVN
        concat_after (bool): whether to concat attention layer's input
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        activation_type (str): type of activation type.
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 input_layer: str = 'conv2d',
                 pos_enc_layer_type: str = 'abs_pos',
                 normalize_before: bool = True,
                 feature_norm: bool = True,
                 concat_after: bool = False,
                 activation_type: str = 'relu',
                 global_cmvn: mindspore.nn.Cell = None,
                 compute_type=mstype.float32):
        """Construct TransformerEncoder."""
        super().__init__(input_size, output_size, positional_dropout_rate, input_layer, pos_enc_layer_type,
                         normalize_before, feature_norm, global_cmvn, compute_type)
        activation = get_activation(activation_type)

        self.encoders = nn.CellList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    compute_type,
                ),
                PositionwiseFeedForward(
                    output_size,
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

    def construct(self,
                  xs: mindspore.Tensor,
                  masks: mindspore.Tensor,
                  xs_chunk_masks: mindspore.Tensor = None) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            masks: masks for the input xs ()
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: mindspore.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        if self.global_cmvn:
            xs = self.global_cmvn(xs)

        # masks is subsampled to (B, 1, T/subsample_rate)
        xs, _ = self.embed(xs)
        for layer in self.encoders:
            xs, xs_chunk_masks = layer(xs, xs_chunk_masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


class ConformerEncoder(BaseEncoder):
    """Transformer encoder module.

    Args:
        input_size (int): input dim
        output_size (int): dimension of attention
        attention_heads (int): the number of heads of multi head attention
        linear_units (int): the hidden units number of position-wise feed
            forward
        num_blocks (int): the number of decoder blocks
        dropout_rate (float): dropout rate
        attention_dropout_rate (float): dropout rate in attention
        positional_dropout_rate (float): dropout rate after adding
            positional encoding
        input_layer (str): input layer type.
            optional [linear, conv2d, conv2d6, conv2d8]
        pos_enc_layer_type (str): Encoder positional encoding layer type.
            opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        feature_norm (bool): whether do feature norm to input features, like CMVN
        concat_after (bool): whether to concat attention layer's input
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        activation_type (str): type of activation type.
        cnn_module_kernel (int): kernel size for CNN module
        cnn_module_norm (str): normalize type for CNN module, batch norm or layer norm.
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 input_layer: str = 'conv2d',
                 pos_enc_layer_type: str = 'rel_pos',
                 normalize_before: bool = True,
                 feature_norm: bool = True,
                 concat_after: bool = False,
                 activation_type: str = 'swish',
                 cnn_module_kernel: int = 15,
                 cnn_module_norm: str = 'batch_norm',
                 global_cmvn: mindspore.nn.Cell = None,
                 compute_type=mstype.float32):
        """Construct ConformerEncoder."""
        super().__init__(input_size, output_size, positional_dropout_rate, input_layer, pos_enc_layer_type,
                         normalize_before, feature_norm, global_cmvn, compute_type)

        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type != 'rel_pos':
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            compute_type,
        )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            compute_type,
        )

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, 1, True, compute_type)

        self.encoders = nn.CellList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args),  # Macron-Net style
                convolution_layer(*convolution_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
                compute_type,
            ) for _ in range(num_blocks)
        ])
