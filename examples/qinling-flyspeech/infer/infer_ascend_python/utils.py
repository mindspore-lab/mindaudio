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
"""Define methods."""
from typing import List

import numpy as np


def load_language_dict(dict_file):
    """Load dict for ASR."""
    char_dict = {}
    with open(dict_file, "r") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    sos = len(char_dict) - 1
    eos = len(char_dict) - 1
    vocab_size = len(char_dict)
    return sos, eos, vocab_size, char_dict


def get_padding_length(length, frame_bucket_limits):
    """Get the padding length through th bucket limitation."""
    for limit in frame_bucket_limits:
        if limit > length:
            return limit
    return frame_bucket_limits[-1]


def make_pad_mask(lengths: List[int], max_len: int = 0):
    """Make mask containing indices of padded part."""
    batch_size = int(len(lengths))
    max_len = max_len if max_len > 0 else max(lengths)
    seq_range = np.expand_dims(np.arange(0, max_len), 0)
    seq_range_expand = np.tile(seq_range, (batch_size, 1))
    seq_length_expand = np.expand_dims(lengths, -1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def subsequent_mask(size: int):
    """Create mask for subsequent steps (size, size)."""
    seq_range = np.arange(size)
    seq_range_expand = np.tile(seq_range, (size, 1))
    seq_length_expand = np.expand_dims(seq_range, -1)
    mask = seq_range_expand <= seq_length_expand
    return mask


def mask_finished_scores(score, end_flag):
    """If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (mindspore.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (mindspore.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        mindspore.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.shape[-1]
    zero_mask = np.zeros_like(end_flag)
    if beam_size > 1:
        unfinished = np.concatenate(
            (zero_mask, np.tile(end_flag, (1, beam_size - 1))), axis=1
        )
        finished = np.concatenate(
            (end_flag, np.tile(zero_mask, (1, beam_size - 1))), axis=1
        )
    else:
        unfinished = zero_mask
        finished = end_flag
    score = np.add(score, np.multiply(unfinished, -10000.0))
    score = np.multiply(score, (1 - finished))

    return score


def mask_finished_preds(pred, end_flag, eos):
    """If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (mindspore.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (mindspore.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        mindspore.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.shape[-1]
    finished = np.tile(end_flag, (1, beam_size)).astype(np.int32)
    pred = pred * (1 - finished) + eos * finished
    return pred


def topk_fun(logits, topk=5):
    """Get topk."""
    batch_size, _ = logits.shape
    value = []
    index = []
    for i in range(batch_size):
        target_column = logits[i].tolist()
        sorted_array = [(k, v) for k, v in enumerate(target_column)]
        sorted_array.sort(key=lambda x: x[1], reverse=True)
        topk_array = sorted_array[:topk]
        index_tmp, value_tmp = zip(*topk_array)
        value.append(value_tmp)
        index.append(index_tmp)
    return np.array(value), np.array(index)
