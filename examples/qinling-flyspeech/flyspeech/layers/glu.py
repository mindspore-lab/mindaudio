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
"""Definition of GLU activation function."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class GLU(nn.Cell):
    """The gated linear unit.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>

    Args:
        num_out (int): the number of out. Default: 2
        dim (int): dimension on which to split the input. Default: 1
    """

    def __init__(self, dim: int = 1, num_out: int = 2):
        super().__init__()
        self.split = ops.Split(dim, num_out)
        self.mul = ops.Mul()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        out, gate = self.split(x)
        out = self.mul(out, self.sigmoid(gate))

        return out
