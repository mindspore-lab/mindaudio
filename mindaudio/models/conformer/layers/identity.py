"""Definition of an identity layer."""

import mindspore.nn as nn
import mindspore.ops as ops


class Identity(nn.Cell):
    """Definition of an identity layer. It will return an identity result
    during the forward process, while multiply a scaled gradient during the
    backward process.

    Args:
        coef (float): Scale for the gradient during the backward process.
    """

    def __init__(self, coef=0.1):
        super(Identity, self).__init__()
        self.identity = ops.Identity()
        self.coef = coef

    def construct(self, x):
        return self.identity(x)

    def bprop(self, x, out, d_out):  # pylint: disable=W0613
        return (d_out * self.coef,)
