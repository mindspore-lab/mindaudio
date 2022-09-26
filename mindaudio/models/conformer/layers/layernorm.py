"""A self-defined layer norm layer."""

import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore.ops as ops
import mindspore.nn as nn


class LayerNorm(nn.Cell):
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
            raise TypeError("The type of parameter 'param_init_type' should in [float32, float16], "
                            'but got the type : {}.'.format(type(param_init_type)))

        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type),
                               name='gamma',
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type),
                              name='beta',
                              parallel_optimizer=False)
        self.mean = ops.ReduceMean(keep_dims=True)
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()
        self.sub1 = ops.Sub()
        self.sub2 = ops.Sub()
        self.add = ops.Add()
        self.eps = epsilon
        self.mul = ops.Mul()
        self.add2 = ops.Add()
        self.real_div = ops.RealDiv()

    def construct(self, x):
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output
