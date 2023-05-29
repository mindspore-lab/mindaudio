"""Self-defined Conv1d layer."""

import math

import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import (HeUniform, Uniform,
                                          _calculate_correct_fan, initializer)
from mindspore.nn.cell import Cell


class Conv2d(Cell):
    """A self-defined 2d convolution layer.

    Args:
        in_channel (int): The channel number of the input tensor of the Conv1d layer.
        out_channel (int): The channel number of the output tensor of the Conv1d layer.
        kernel_size (int): Specifies the width of the 1D convolution kernel.
        stride (int): The movement stride of the 1D convolution kernel.
        has_bias (bool): Whether the Conv1d layer has a bias parameter.
        pad_mode (str): Specifies padding mode, ["same", "valid", "pad"].
        negative_slope (int, float, bool): The negative slope of the rectifier used after this layer.
        mode (str): Either "fan_in" or "fan_out".
        nonlinerity (str): The non-linear function, use only with "relu" or "leaky_relu"
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        has_bias=False,
        pad_mode="valid",
        negative_slope=math.sqrt(5),
        mode="fan_in",
        nonlinerity="leaky_relu",
    ):
        super(Conv2d, self).__init__()
        kaiming_uniform_0 = initializer(
            HeUniform(
                negative_slope=negative_slope, mode=mode, nonlinearity=nonlinerity
            ),
            (out_channel, in_channel, kernel_size, kernel_size),
        )
        fan_in = _calculate_correct_fan(
            (out_channel, in_channel, kernel_size, kernel_size), mode=mode
        )
        scale = 1 / math.sqrt(fan_in)
        bias_init_0 = initializer(Uniform(scale), [out_channel], mindspore.float32)
        self.conv2d = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            has_bias=has_bias,
            pad_mode=pad_mode,
            weight_init=kaiming_uniform_0,
            bias_init=bias_init_0,
        )

    def construct(self, x):
        out = self.conv2d(x)
        return out
