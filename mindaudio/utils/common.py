# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

from typing import Tuple, List

import numpy as np

import mindspore
import mindspore.nn as nn

IGNORE_ID = -1


def pad_sequence(
        sequences: List[np.ndarray],
        batch_first=True,
        padding_value: int = 0,
        padding_max_len: int = None,
        atype=np.int32,
) -> np.ndarray:
    """[summary]

    Args:
        sequences (List[np.ndarray]): [description]
        batch_first (bool, optional): [description]. Defaults to True.
        padding_value (int, optional): [description]. Defaults to 0.
        padding_max_len (int, optional): [description]. Defaults to None.
        atype ([type], optional): [description]. Defaults to np.int32.

    Returns:
        np.ndarray: [description]
    """
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]

    if padding_max_len is not None:
        max_len = padding_max_len
    else:
        max_len = max([s.shape[0] for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_sequences = np.full(out_dims, fill_value=padding_value).astype(atype)

    for i, seq in enumerate(sequences):
        length = seq.shape[0] if seq.shape[0] <= max_len else max_len
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_sequences[i, :length, ...] = seq[:length]
        else:
            out_sequences[:length, i, ...] = seq[:length]

    return out_sequences

def add_sos_eos(ys: List[np.ndarray],
                sos: int = 0,
                eos: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Add <sos> and <eos> labels. For generating the decoder input and output.

    Args:
        ys (List[np.ndarray]): list of target sequences
        sos (int): index of <sos>
        eos (int): index of <eos>

    Returns:
        ys_in (List[np.ndarray])
        ys_out (List[np.ndarray])

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ys_pad
        [array([ 1,  2,  3,  4,  5]),
         array([ 4,  5,  6]),
         array([ 7,  8,  9])]
        >>> ys_in, ys_out = add_sos_eos(ys_pad, sos_id, eos_id)
        >>> ys_in
        [array([10,  1,  2,  3,  4,  5]),
         array([10,  4,  5,  6]),
         array([10,  7,  8,  9])]
        >>> ys_out
        [array([ 1,  2,  3,  4,  5, 11]),
         array([ 4,  5,  6, 11]),
         array([ 7,  8,  9, 11])]
    """
    ys_in = [np.concatenate(([sos], y), axis=0) for y in ys]
    ys_out = [np.concatenate((y, [eos]), axis=0) for y in ys]
    return ys_in, ys_out


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from src.layers.swish import Swish

    activation_funcs = {
        "tanh": mindspore.nn.Tanh,
        "relu": mindspore.nn.ReLU,
        "swish": Swish,
        "gelu": mindspore.nn.GELU,
    }

    return activation_funcs[act]()


def get_subsample(config):
    input_layer = config["encoder_conf"]["input_layer"]
    assert input_layer in ["conv2d", "conv2d6", "conv2d8"]
    if input_layer == "conv2d":
        return 4
    if input_layer == "conv2d6":
        return 6
    return 8


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float("inf") for a in args):
        return -float("inf")
    a_max = max(args)
    lsp = np.log(sum(np.exp(a - a_max) for a in args))
    return a_max + lsp


def get_parameter_numel(net: nn.Cell):
    num = (np.array([np.prod(item.shape)
                     for item in net.get_parameters()]).sum() / 1024 / 1024)
    return str(num)[:5] + "M"


def set_weight_decay(params, weight_decay=1e-2):
    """
    Set weight decay coefficient, zero for bias and layernorm, default 1e-2 for rest
    """
    decay_filter = lambda x: 'layernorm' not in x.name.lower(
    ) and "bias" not in x.name.lower()
    decay_params = list(filter(decay_filter, params))
    other_params = list(filter(lambda x: not decay_filter(x), params))
    group_params = [{
        'params': decay_params,
        'weight_decay': weight_decay
    }, {
        'params': other_params,
        'weight_decay': 0.0
    }, {
        'order_params': params
    }]
    return group_params


def get_feat_extract_output_lengths(input_length, kernel_size, stride):
    """
    get seqs length after cnns module downsampling.
    """
    len_ds = input_length
    for i in range(len(kernel_size)):
        len_ds = (len_ds - kernel_size[i]) // stride[i] + 1
    return len_ds
