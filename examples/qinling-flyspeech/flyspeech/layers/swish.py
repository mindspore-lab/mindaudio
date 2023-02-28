# Copyright (c) 2020 Johns Hopkins University (Shinji Watanabe)
#               2020 Northwestern Polytechnical University (Pengcheng Guo)
#               2020 Mobvoi Inc (Binbin Zhang)
# 2022.07 - Modified code to support Mindspore
#           Huawei Technologies Co., Ltd
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
