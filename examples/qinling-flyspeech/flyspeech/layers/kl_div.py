# Copyright 2020 Huawei Technologies Co., Ltd
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
"""The Kullback-Leibler divergence loss."""

import mindspore
import mindspore.ops as ops


class KLDivLoss(mindspore.nn.Cell):
    """Construct an KLDivLoss module."""

    def __init__(self):
        super(KLDivLoss, self).__init__()
        self.log = ops.Log()
        self.mul = ops.Mul()

    def construct(self, p: mindspore.Tensor, q: mindspore.Tensor) -> mindspore.Tensor:
        log_term = self.log(q) - p
        kl_div = self.mul(q, log_term)
        return kl_div
