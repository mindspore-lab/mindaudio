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
"""Positionwise feed forward layer definition."""
import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn

from flyspeech.layers.dense import Dense


class PositionwiseFeedForward(nn.Cell):
    """Positionwise feed forward layer.

    FeedForward are applied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimension.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation function.
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: nn.Cell,
                 compute_type=mstype.float32):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Dense(idim, hidden_units).to_float(compute_type)
        self.activation = activation
        self.dropout = nn.Dropout(1 - dropout_rate)
        self.w_2 = Dense(hidden_units, idim).to_float(compute_type)

    def construct(self, xs: mindspore.Tensor) -> mindspore.Tensor:
        """Forward function.

        Args:
            xs (mindspore.Tensor): Input tensor (B, L, D)
        Returns:
            mindspore.Tensor: Output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
