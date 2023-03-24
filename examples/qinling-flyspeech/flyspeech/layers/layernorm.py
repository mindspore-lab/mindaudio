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
"""A self-defined layer norm layer."""

import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P


class LayerNorm(Cell):
    """A self-defined layer norm operation using reduce sum and reduce mean.

    Args:
        normalized_shape (tuple): The shape of the input tensor.
        epsilon (float): The epsilon value of the denominator. Default 1e-5.
        param_init_type: The param init type.
    Inputs:
        x (mindspore.Tensor): `(batch, seq_length, hidden_size)`.

    Outputs:
        mindspore.Tensor: `(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, epsilon=1e-5, param_init_type=mstype.float32):
        super(LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16]:
            raise TypeError(
                "The type of parameter 'param_init_type' should in [float32, float16], "
                "but got the type : {}.".format(type(param_init_type))
            )

        self.gamma = Parameter(
            initializer("ones", normalized_shape, param_init_type),
            name="gamma",
            parallel_optimizer=False,
        )
        self.beta = Parameter(
            initializer("zeros", normalized_shape, param_init_type),
            name="beta",
            parallel_optimizer=False,
        )
        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.sub1 = P.Sub()
        self.sub2 = P.Sub()
        self.add = P.Add()
        self.eps = epsilon
        self.mul = P.Mul()
        self.add2 = P.Add()
        self.real_div = P.RealDiv()

    def construct(self, x):
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output
