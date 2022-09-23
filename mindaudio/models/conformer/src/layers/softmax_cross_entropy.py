"""Definition of sofmax cross entroy loss."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


class SoftmaxCrossEntropyWithLogits(nn.Cell):
    """"Definition of sofmax cross entroy loss."""

    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.neg = ops.Neg()

    def construct(self, logits, label, mask):
        logits = self.log_softmax(logits)
        mask = mask.transpose(1, 0).view(-1)
        numerator = self.neg((logits * label).sum(-1)) * mask
        numerator = numerator.sum()
        denominator = mask.sum() + self.cast(ops.tuple_to_array((1e-5,)), mindspore.float32)
        loss = numerator / denominator
        return loss
