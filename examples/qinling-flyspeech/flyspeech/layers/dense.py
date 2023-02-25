# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Self-defined Dense layer."""

import math

import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import HeUniform, Uniform, _calculate_correct_fan, initializer
from mindspore.nn.cell import Cell


class Dense(Cell):
    """A self-defined dense layer.

    Args:
        in_channel (int): The number of channels in the input space.
        out_channel (int): The number of channels in the output space.
        has_bias (bool): The trainable bias_init parameter.
        activation (nn.Cell): activate function applied to the output of the fully connected layer.
        negative_slope (int, float, bool): The negative slope of the rectifier used after this layer.
        mode (str): Either "fan_in" or "fan_out".
        nonlinerity (str): The non-linear function, use only with "relu" or "leaky_relu"
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 has_bias=True,
                 activation=None,
                 negative_slope=math.sqrt(5),
                 mode='fan_in',
                 nonlinearity='leaky_relu'):
        super(Dense, self).__init__()
        kaiming_uniform_0 = initializer(HeUniform(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinearity),
                                        (out_channel, in_channel))
        bias_init_0 = 'zeros'
        if has_bias:
            fan_in = _calculate_correct_fan((out_channel, in_channel), mode=mode)
            scale = 1 / math.sqrt(fan_in)
            bias_init_0 = initializer(Uniform(scale), [out_channel], mindspore.float32)
        self.dense = nn.Dense(in_channel,
                              out_channel,
                              weight_init=kaiming_uniform_0,
                              bias_init=bias_init_0,
                              has_bias=has_bias,
                              activation=activation)

    def construct(self, x):
        out = self.dense(x)
        return out
