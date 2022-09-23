"""Self-defined Conv1d layer."""

import math

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal, HeUniform, Uniform, _calculate_correct_fan, initializer


class Conv1d(nn.Cell):
    """A self-defined 1d convolution layer.

    Args:
        in_channel (int): The channel number of the input tensor of the Conv1d layer.
        out_channel (int): The channel number of the output tensor of the Conv1d layer.
        kernel_size (int): Specifies the width of the 1D convolution kernel.
        stride (int): The movement stride of the 1D convolution kernel.
        padding (int): The number of padding on both sides of input.
        group (int): Splits filter into groups.
        has_bias (bool): Whether the Conv1d layer has a bias parameter.
        pad_mode (str): Specifies padding mode, ["same", "valid", "pad"].
        negative_slope (int, float, bool): The negative slope of the rectifier used after this layer.
        mode (str): Either "fan_in" or "fan_out".
        nonlinerity (str): The non-linear function, use only with "relu" or "leaky_relu"
        init (str): Parameter initialize type.
        enable_mask_padding_feature (bool): Whether to zero the masked part of input.
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 group=1,
                 has_bias=False,
                 pad_mode='valid',
                 negative_slope=math.sqrt(5),
                 mode='fan_in',
                 nonlinerity='leaky_relu',
                 init='heuniform',
                 enable_mask_padding_feature=True):
        super(Conv1d, self).__init__()
        if init == 'heuniform':
            kaiming_uniform_0 = initializer(
                HeUniform(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinerity),
                (out_channel, in_channel // group, kernel_size))
        else:
            kaiming_uniform_0 = initializer(
                HeNormal(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinerity),
                (out_channel, in_channel // group, kernel_size))
        fan_in = _calculate_correct_fan((out_channel, in_channel // group, kernel_size), mode=mode)
        scale = 1 / math.sqrt(fan_in)
        bias_init_0 = initializer(Uniform(scale), [out_channel], mindspore.float32)
        self.conv1d = nn.Conv1d(in_channel,
                                out_channel,
                                kernel_size,
                                stride=stride,
                                has_bias=has_bias,
                                pad_mode=pad_mode,
                                padding=padding,
                                group=group,
                                weight_init=kaiming_uniform_0,
                                bias_init=bias_init_0)
        self.floor_div = ops.FloorDiv()
        self.tile = ops.Tile()
        self.stride = stride
        self.kernel_size = kernel_size
        self.expand_dims = ops.ExpandDims()
        self.enable_mask_padding_feature = enable_mask_padding_feature

    def construct(self, x, x_len: mindspore.Tensor = None):
        """1d convolution layer."""
        out = self.conv1d(x)

        if not self.enable_mask_padding_feature:
            return out

        bs, _, total_length = out.shape
        valid_length = self.floor_div((x_len - self.kernel_size), self.stride) + 1
        total_length_range = ops.tuple_to_array(ops.make_range(total_length))  #0, 1, 2, 3, ..., total-1
        total_length_range = self.tile(self.expand_dims(total_length_range, 0), (bs, 1))  # bs, total_
        valid_length_range = self.expand_dims(valid_length, -1)  # bs, 1
        valid_mask = (total_length_range < valid_length_range).astype(mindspore.float32)  #bs, seq
        out = out * valid_mask.expand_dims(1)

        return out, valid_length


class Conv2d(nn.Cell):
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

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 has_bias=False,
                 pad_mode='valid',
                 negative_slope=math.sqrt(5),
                 mode='fan_in',
                 nonlinerity='leaky_relu'):
        super(Conv2d, self).__init__()
        kaiming_uniform_0 = initializer(HeUniform(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinerity),
                                        (out_channel, in_channel, kernel_size, kernel_size))
        fan_in = _calculate_correct_fan((out_channel, in_channel, kernel_size, kernel_size), mode=mode)
        scale = 1 / math.sqrt(fan_in)
        bias_init_0 = initializer(Uniform(scale), [out_channel], mindspore.float32)
        self.conv2d = nn.Conv2d(in_channel,
                                out_channel,
                                kernel_size,
                                stride=stride,
                                has_bias=has_bias,
                                pad_mode=pad_mode,
                                weight_init=kaiming_uniform_0,
                                bias_init=bias_init_0)

    def construct(self, x):
        out = self.conv2d(x)
        return out
