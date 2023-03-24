# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Definition of sofmax cross entroy loss."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn.cell import Cell


class SoftmaxCrossEntropyWithLogits(Cell):
    """ "Definition of sofmax cross entroy loss."""

    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.neg = ops.Neg()

    def construct(self, logits, label, mask):
        logits = self.log_softmax(logits)
        mask = mask.transpose(1, 0).view(-1)
        numerator = self.neg((logits * label).sum(-1)) * mask
        numerator = numerator.sum()
        denominator = mask.sum() + self.cast(
            ops.tuple_to_array((1e-5,)), mindspore.float32
        )
        # denominator = mask.sum() + 1e-5
        loss = numerator / denominator
        return loss
