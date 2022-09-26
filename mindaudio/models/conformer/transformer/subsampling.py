# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
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
"""Subsampling layer definition."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from mindaudio.models.conformer.layers.conv import Conv2d
from mindaudio.models.conformer.layers.dense import Dense


class BaseSubsampling(nn.Cell):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> mindspore.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        pos_enc_type (nn.Cell): Type of positional encoding layer.
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self, idim: int, odim: int, pos_enc_class: nn.Cell, compute_type=mindspore.float32):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = nn.SequentialCell(
            Conv2d(
                in_channel=1,
                out_channel=odim,
                kernel_size=3,
                stride=2,
                has_bias=True,
                pad_mode='valid'
            ),
            nn.ReLU(),
            Conv2d(
                in_channel=odim,
                out_channel=odim,
                kernel_size=3,
                stride=2,
                has_bias=True,
                pad_mode='valid'
            ),
            nn.ReLU(),
        ).to_float(compute_type)
        self.compute_type = compute_type
        self.out = Dense(odim * (((idim-1) // 2 - 1) // 2), odim).to_float(compute_type)
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6
        self.expanddims = ops.ExpandDims()
        self.cast = ops.Cast()

    def construct(self, x: mindspore.Tensor, offset: int = 0) -> mindspore.Tensor:
        """Subsample x.

        Args:
            x (minspore.Tensor): Input tensor (#batch, time, idim).
            x_mask (minspore.Tensor): Input mask (#batch, 1, time).

        Returns:
            minspore.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            minspore.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            minspore.Tensor: positional encoding
        """
        x = self.expanddims(x, 1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = self.out(x.transpose(0, 2, 1, 3).view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb
