"""ConvolutionModule definition."""

from typing import Tuple

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from .conv1d import Conv1d
from .glu import GLU
from .layernorm import LayerNorm


class ConvolutionModule(nn.Cell):
    """Construct an ConvolutionModule object.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.
        activation (nn.Cell): Activation function of CNN module.
        norm (str): Normalize type of CNN module, batch norm or layer norm.
        glu_dim (int): Dimension of GLU activation function.
        bias (bool): Whether use bias for CNN layer.
        compute_type (bool): Whether use mix precision computation.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        activation: nn.Cell,
        norm: str = "batch_norm",
        glu_dim: int = 1,
        bias: bool = True,
        compute_type=mindspore.float32,
    ):
        super().__init__()
        self.pointwise_conv1 = Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=bias,
            pad_mode="valid",
            enable_mask_padding_feature=False,
        ).to_float(compute_type)
        self.depthwise_conv = Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            group=channels,
            has_bias=bias,
            pad_mode="pad",
            enable_mask_padding_feature=False,
        ).to_float(compute_type)

        assert norm in ["batch_norm", "layer_norm"]
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = LayerNorm(channels)
        self.pointwise_conv2 = Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=bias,
            pad_mode="valid",
            enable_mask_padding_feature=False,
        ).to_float(compute_type)
        self.channels = channels
        self.activation = activation
        self.glu = GLU(dim=glu_dim)
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()

    def construct(
        self, x: mindspore.Tensor, mask: mindspore.Tensor = None
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Compute convolution module.

        Args:
            x (mask.Tensor): Input tensor (#batch, time, channels).
            mask (mindspore.Tensor): Mask (#batch, 1, time)
        Returns:
            mindspore.Tensor: Output tensor (#batch, time, channels).
        """
        x = x.transpose(0, 2, 1)  # (batch, channels, time)

        # mask padding part
        if mask is not None:
            x = x * mask

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2 * channels, time)
        x = self.glu(x)  # (batch, channels, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)  # (batch, channels, time)

        if self.use_layer_norm:
            # LayerNorm compute the mean and std over the last dimension,
            # thus change the channel dimension to the last dimension
            x = x.transpose(0, 2, 1)
            x = self.activation(self.norm(x))
            x = x.transpose(0, 2, 1)
        else:
            # x = self.cast(x, mstype.float32)
            x = x.transpose(0, 2, 1)
            batch, length, channel = x.shape
            x = self.reshape(x, (batch * length, channel))
            x = self.activation(self.norm(x))
            x = self.reshape(x, (batch, length, channel))
            x = x.transpose(0, 2, 1)
            # x = self.cast(x, mstype.float16)

        x = self.pointwise_conv2(x)

        # mask padding part
        if mask is not None:
            x = x * mask

        return x.transpose(0, 2, 1)
