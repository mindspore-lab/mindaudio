"""Positonal Encoding Module."""

import math
from typing import Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.operations as ops
import numpy as np
from mindspore.common.tensor import Tensor

from .conv1d import Conv1d


class PositionalEncoding(nn.Cell):
    """Positional encoding.

    Args:
    d_model (int): embedding dim
    dropout_rate (float): dropout rate
    max_len (int): maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = Tensor([math.sqrt(self.d_model)], dtype=mstype.float32)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = np.zeros((self.max_len, self.d_model))
        position = np.expand_dims(np.arange(0, self.max_len, dtype=np.float32), 1)
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Tensor(np.expand_dims(self.pe, 0), mstype.float32)

    def construct(
        self, x: mindspore.Tensor, offset: int = 0
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Compute positional encoding.

        Args:
            x (minspore.Tensor): Input tensor (batch, time, `*`).
        Returns:
            minspore.Tensor: Encoded tensor (batch, time, `*`).
            minspore.Tensor: Positional embedding tensor (1, time, `*`).
        """
        pos_emb = self.pe[:, offset : offset + x.shape[1]]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> mindspore.Tensor:
        return self.dropout(self.pe[:, offset : offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.

    See : Appendix B in https://arxiv.org/abs/1901.02860/
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def construct(
        self, x: mindspore.Tensor, offset: int = 0
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Compute positional encoding.

        Args:
            x (minspore.Tensor): Input tensor (batch, time, `*`).
        Returns:
            minspore.Tensor: Encoded tensor (batch, time, `*`).
            minspore.Tensor: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.pe[:, offset : offset + x.shape[1]]
        return self.dropout(x), self.dropout(pos_emb)


class ConvPositionalEncoding(nn.Cell):
    """ConvPositionalEncoding.

    Args:
        d_model (int): Model embedding dimension
        dropout_rate (float): Dropout rate
        max_len (int): Maximum input length
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an convolution PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = Tensor([math.sqrt(self.d_model)], dtype=mstype.float32)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = np.zeros((self.max_len, self.d_model))
        self.pe = Tensor(np.expand_dims(self.pe, 0), mstype.float32)
        self.pos_conv = Conv1d(
            d_model,
            d_model,
            kernel_size=128,
            pad_mode="pad",
            padding=64,
            has_bias=True,
            enable_mask_padding_feature=False,
        )
        self.stride_slice = ops.StridedSlice()
        self.gelu = nn.GELU()

    def construct(
        self, x: mindspore.Tensor, offset: int = 0
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Compute positional encoding.

        Args:
            x (minspore.Tensor): Input tensor (batch, time, `*`).
        Returns:
            minspore.Tensor: Encoded tensor (batch, time, `*`).
            minspore.Tensor: Positional embedding tensor (1, time, `*`).
        """
        b, t, c = x.shape  # B T C
        x_pos = x.transpose(0, 2, 1)  # B C T
        x_pos = self.pos_conv(x_pos)
        x_pos = x_pos.transpose(0, 2, 1)  # B T C
        x_pos = self.stride_slice(x_pos, (0, 0, 0), (b, t, c), (1, 1, 1))
        x_pos = self.gelu(x_pos)
        x_pos = x + x_pos  # B T C
        pos_emb = self.pe[:, offset : offset + x.shape[1]]
        return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(nn.Cell):
    """No position encoding
    Args:
    d_model (int): Model embedding dimension
    dropout_rate (float): Dropout rate
    """

    def __init__(self, d_model: int, dropout_rate: float):
        super(NoPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)
        self.zeros = ops.Zeros()

    def construct(self, x: mindspore.Tensor, offset: int = 0):  # pylint: disable=W0613
        """Just return zero vector for interface compatibility
        Args:
            x (minspore.Tensor): Input tensor (batch, time, `*`).
        Returns:
            minspore.Tensor: Encoded tensor (batch, time, `*`).
            minspore.Tensor: Positional embedding tensor (1, time, `*`).
        """
        pos_emb = self.zeros((1, x.shape[1], self.d_model), mstype.float32)
        return self.dropout(x), pos_emb
