"""Positionwise feed forward layer definition."""
import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from layers.dense import Dense


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

    def __init__(
        self,
        idim: int,
        hidden_units: int,
        dropout_rate: float,
        activation: nn.Cell,
        compute_type=mstype.float32,
    ):
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
