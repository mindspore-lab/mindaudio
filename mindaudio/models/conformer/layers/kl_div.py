"""The Kullback-Leibler divergence loss."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class KLDivLoss(nn.Cell):
    """Construct an KLDivLoss module."""

    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.log = ops.Log()
        self.mul = ops.Mul()

    def construct(self, p: mindspore.Tensor, q: mindspore.Tensor) -> mindspore.Tensor:
        log_term = self.log(q) - p
        kl_div = self.mul(q, log_term)
        return kl_div
