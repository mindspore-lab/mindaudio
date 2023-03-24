# Copyright (c) 2019 Shigeki Karita
# 2022.07 - Modified the code to support Mindspore
#           Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Label smoothing module."""

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from flyspeech.layers.kl_div import KLDivLoss
from mindspore.common.tensor import Tensor


class LabelSmoothingLoss(nn.Cell):
    """Label-smoothing loss.

    In a standard CE loss, the label's data distribution is:
    [0,1,2] ->
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    In the smoothing version CE Loss,some probabilities
    are taken from the true label prob (1.0) and are divided
    among other labels.

    e.g.
    smoothing=0.1
    [0,1,2] ->
    [
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ]

    Args:
        size (int): the number of class
        padding_idx (int): padding class id which will be ignored for loss
        smoothing (float): smoothing rate (0.0 means the conventional CE)
        normalize_length (bool):
            normalize loss by sequence length if True
            normalize loss by batch size if False
    """

    def __init__(
        self,
        size: int,
        padding_idx: int,
        smoothing: float,
        normalize_length: bool = False,
    ):
        """Construct an LabelSmoothingLoss object."""
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = KLDivLoss()
        self.padding_idx = padding_idx
        self.on_value = Tensor([1.0 - smoothing], dtype=mstype.float32)
        self.off_value = Tensor([smoothing / (size - 1)], dtype=mstype.float32)
        self.size = size  # vocab size
        self.normalize_length = normalize_length
        self.log_softmax = nn.LogSoftmax(1)
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.mul = ops.Mul()
        self.onehot = ops.OneHot(axis=-1)

    def construct(
        self,
        x: mindspore.Tensor,
        target: mindspore.Tensor,
        target_masks: mindspore.Tensor,
    ) -> mindspore.Tensor:
        """Compute loss between x and target.

        The model outputs and data labels tensors are flatten to
        (batch*seqlen, class) shape and a mask is applied to the
        padding part which should not be calculated for loss.

        Args:
            x (mindspore.Tensor): prediction (batch, seqlen, class)
            target (mindspore.Tensor):
                target sequence masked with self.padding_id (batch, seqlen)
            target_masks (mindspore.Tensor): target sequence masks to indicate
                the padding part

        Returns:
            mindspore.Tensor: The KL loss, scalar float value
        """
        batch_size = x.shape[0]
        x = x.view(-1, self.size)
        target = target.view(-1)
        target_masks = target_masks.view(-1)
        target_zeropad = self.cast(
            self.mul(target, target_masks), mstype.int32
        )  # avoid -1 index
        total = target_masks.sum()
        denom = total if self.normalize_length else batch_size
        true_dist = self.onehot(
            target_zeropad, self.size, self.on_value, self.off_value
        )

        kl = self.criterion(self.log_softmax(x), true_dist)
        # mask the loss of padded part
        kl = self.mul(kl, self.expand_dims(target_masks, 1))

        return kl.sum() / denom
