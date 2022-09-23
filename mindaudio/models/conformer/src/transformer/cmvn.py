# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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
"""cepstral mean and variance normalization definition."""

import mindspore
import mindspore.nn as nn


class GlobalCMVN(nn.Cell):
    """cmvn definition.

    Args:
        mean (mindspore.Tensor): mean stats
        istd (mindspore.Tensor): inverse std, std which is 1.0 / std
        norm_var (bool): Whether variance normalization is performed, default: True
    """

    def __init__(self, mean: mindspore.Tensor, istd: mindspore.Tensor, norm_var: bool = True):
        """Construct an CMVN object."""
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        self.mean = mean
        self.istd = istd

    def construct(self, x: mindspore.Tensor):
        """the calculation process for cmvn.

        Args:
            x (mindspore.Tensor): (batch, max_len, feat_dim)
        Returns:
            (mindspore.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x
