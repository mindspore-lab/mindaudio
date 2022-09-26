# Copyright (c) Facebook, Inc. and its affiliates.
# 2022.07 - Modified the code to support Mindspore
#           Huawei Technologies Co., Ltd
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
"""Definition of vector quantizer for wav2vec 2.0 model."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
# pylint: disable=C0412
from mindspore import Tensor
from mindspore.common.initializer import Normal, initializer
from mindspore.ops import stop_gradient


class GumbelSoftmax(nn.Cell):
    """Definition of Gumbel-Softmax function."""

    def __init__(self):
        super().__init__()
        self.ops_softmax = ops.Softmax()
        self.ops_log = ops.Log()
        self.ops_uniform = ops.UniformReal()
        self.zeros_like = ops.ZerosLike()
        self.ones = ops.Ones()
        self.concat = ops.Concat(-1)
        self.expand_dims = ops.ExpandDims()
        self.update_scatter = ops.TensorScatterUpdate()

    def construct(self, logits, tau=1.0, hard=False):
        """Construct a Gumbel-Softmax function."""
        u = self.ops_uniform(logits.shape)
        y = logits - self.ops_log(-self.ops_log(u + 1e-10) + 1e-10)
        y_soft = self.ops_softmax(y / tau)
        y_soft_tmp = self.ops_softmax(y / tau)
        y_soft_tmp = stop_gradient(y_soft_tmp)
        if hard:
            location = y_soft.argmax(axis=-1)
            location = self.expand_dims(location, -1)
            index = ops.tuple_to_array(ops.make_range(location.shape[0]))
            index = self.expand_dims(index, -1)
            indices = self.concat((index, location))
            y_hard = self.zeros_like(logits)
            update_value = self.ones(y_hard.shape[0], mindspore.float16)
            y_hard = self.update_scatter(y_hard, indices, update_value)
            y_hard = stop_gradient(y_hard)
            ret = y_hard - y_soft_tmp + y_soft
        else:
            ret = y_soft
        return ret


class GumbelVectorQuantizer(nn.Cell):
    """Vector quantization using gumbel softmax.

    Args:
        dim (int): Input dimension (channels).
        num_vars (int): Number of quantized vectors per group.
        temp (List[float]): Temperature for training, List[start, stop, decay factor].
        groups (int): Number of groups for vector quantization.
        vq_dim (int): Dimensionality of the resulting quantized vector.
    """

    def __init__(self, dim, num_vars, temp, groups, vq_dim):
        super().__init__()

        self.gumbel_softmax = GumbelSoftmax()

        self.groups = groups
        self.input_dim = dim
        self.num_vars = num_vars

        assert (vq_dim % groups == 0), f'dim {vq_dim} must be divisible by groups {groups} for concatenation'

        var_dim = vq_dim // groups
        num_groups = groups
        self.vars = mindspore.Parameter(
            Tensor(np.random.uniform(0, 1, (1, num_groups * num_vars, var_dim)), dtype=mindspore.float32))

        self.weight_proj = nn.Dense(self.input_dim,
                                    self.groups * self.num_vars,
                                    has_bias=True,
                                    weight_init=initializer(Normal(sigma=1),
                                                            [self.groups * self.num_vars, self.input_dim])).to_float(
                                                                mindspore.float16)
        self.max_temp, self.min_temp, self.temp_decay = temp

        # ops for graph mode definition
        self.ops_scatternd = ops.ScatterNd()
        self.ops_argmax = ops.Argmax(-1)
        self.ops_exp = ops.Exp()
        self.ops_log = ops.Log()
        self.ops_softmax = ops.Softmax()
        self.ops_expand = ops.ExpandDims()
        self.ops_pow = ops.Pow()
        self.ops_max = ops.Maximum()
        self.ops_concat = ops.Concat(axis=1)

    def construct(self, x, mask_valid_index, num_updates):
        """Construct a GumbelVectorQuantizer function."""
        b, t, d = x.shape
        x = x.view(-1, d)

        x = self.weight_proj(x)

        x = x.view(b * t * self.groups, -1)

        # prob perplexity
        x_ = x.reshape(b * t, self.groups, -1)

        probs = self.ops_softmax(x_)
        masked_probs = probs * mask_valid_index.view(-1, 1, 1)  # b*t, 1, 1
        avg_probs = masked_probs.sum(axis=0) / (mask_valid_index.view(-1)).sum()
        prob_perplexity = self.ops_exp(-(avg_probs * self.ops_log(avg_probs + 1e-7)).sum(axis=-1)).sum()

        curr_temp = self.ops_max(self.max_temp * self.ops_pow(self.temp_decay, num_updates), self.min_temp)
        x = self.gumbel_softmax(x, tau=curr_temp, hard=True)
        x = x.view(b * t, -1)
        x = self.ops_expand(x, -1) * self.vars
        x = x.view(b * t, self.groups, self.num_vars, -1).sum(axis=-2)
        x = x.view(b, t, -1)
        return x, prob_perplexity, curr_temp
