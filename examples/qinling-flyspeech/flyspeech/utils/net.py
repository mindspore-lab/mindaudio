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
"""Define net-related methods."""
import mindspore
import numpy as np


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from flyspeech.layers.swish import Swish

    activation_funcs = {
        "tanh": mindspore.nn.Tanh,
        "relu": mindspore.nn.ReLU,
        "swish": Swish,
        "gelu": mindspore.nn.GELU,
    }

    return activation_funcs[act]()


def get_parameter_numel(net: mindspore.nn.Cell):
    num = (
        np.array([np.prod(item.shape) for item in net.get_parameters()]).sum()
        / 1024
        / 1024
    )
    return str(num)[:5] + "M"
