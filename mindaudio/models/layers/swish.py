"""Swish() activation function for Conformer."""

import mindspore
import mindspore.ops as ops


class Swish(mindspore.nn.Cell):
    """Construct an Swish activation function object."""

    def __init__(self):
        super().__init__()
        self.sigmoid = ops.Sigmoid()

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """Return Swish activation function."""
        return x * self.sigmoid(x)
