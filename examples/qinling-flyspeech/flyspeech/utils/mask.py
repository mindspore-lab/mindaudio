# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
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
"""mask function."""

from typing import List

import mindspore
import mindspore.ops as ops
import numpy as np

TILE_FUNC = ops.Tile()
CAT_FUNC = ops.Concat(axis=1)
MUL_FUNC = ops.Mul()
ADD_FUNC = ops.Add()
EQUAL_FUNC = ops.Equal()
ZEROSLIKE_FUNC = ops.ZerosLike()
CAST_FUNC = ops.Cast()
NEG_INF = mindspore.Tensor([-10000.0], dtype=mindspore.float32)


def subsequent_mask(size: int):
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    Args:
        size (int): size of mask

    Returns:
        np.ndarray: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    seq_range = np.arange(size)
    seq_range_expand = np.tile(seq_range, (size, 1))
    seq_length_expand = np.expand_dims(seq_range, -1)
    mask = seq_range_expand <= seq_length_expand
    return mask


def make_pad_mask(lengths: List[int], max_len: int = 0):
    """Make mask containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (List[int]): Batch of lengths (B,).
    Returns:
        np.ndarray: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = int(len(lengths))
    max_len = max_len if max_len > 0 else max(lengths)
    seq_range = np.expand_dims(np.arange(0, max_len), 0)
    seq_range_expand = np.tile(seq_range, (batch_size, 1))
    seq_length_expand = np.expand_dims(lengths, -1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: List[int], max_len: int = 0):
    """Make mask containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (List[int]): Batch of lengths (B,).
    Returns:
        np.ndarray: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths, max_len)


def mask_finished_scores(score: mindspore.Tensor, end_flag: mindspore.Tensor):
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
    zero_mask = ZEROSLIKE_FUNC(end_flag)
    if beam_size > 1:
        unfinished = CAT_FUNC((zero_mask, TILE_FUNC(end_flag, (1, beam_size - 1))))
        finished = CAT_FUNC((end_flag, TILE_FUNC(zero_mask, (1, beam_size - 1))))
    else:
        unfinished = zero_mask
        finished = end_flag
    score = ADD_FUNC(score, MUL_FUNC(unfinished, NEG_INF))
    score = MUL_FUNC(score, (1 - finished))

    return score


def mask_finished_preds(
    pred: mindspore.Tensor, end_flag: mindspore.Tensor, eos: int
) -> mindspore.Tensor:
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
    finished = CAST_FUNC(TILE_FUNC(end_flag, (1, beam_size)), mindspore.int32)
    pred = pred * (1 - finished) + eos * finished
    return pred


def compute_mask_indices(shape, mask_prob, mask_length) -> np.ndarray:
    """compute mask indices."""
    b, t = shape
    mask = np.full((b, t), False)
    n_mask = int(mask_prob * t / float(mask_length) + 0.35)
    for i in range(b):
        ti = t
        span = ti // n_mask
        for j in range(n_mask):
            # non-overlaped masking
            start = j * span + np.random.randint(span - mask_length)
            mask[i][start : start + mask_length] = True
    return mask


def compute_mask_indices2(shape, padding_mask, mask_prob, mask_length) -> np.ndarray:
    """compute mask indices2."""
    b, t = shape
    mask = np.full((b, t), False)
    mask_valid = np.full((b, t), False)
    n_mask = int(mask_prob * t / float(mask_length) + 0.35)
    for i in range(b):
        real_wav_len = t - padding_mask[i].astype(int).sum().item()
        ti = t
        span = ti // n_mask
        for j in range(n_mask):
            # non-overlaped masking
            start = j * span + np.random.randint(span - mask_length)
            mask[i][start : start + mask_length] = True
        mask_valid[i][:real_wav_len] = True
    return mask, mask_valid


def subsequent_chunk_mask(size: int, chunk_size: int, num_left_chunks: int = -1):
    """Create mask for subsequent steps (size, size) with chunk size, this is
    for streaming encoder.

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks

    Returns:
        numpy.array: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = np.zeros((size, size), dtype=np.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(
    xs_len,
    masks,
    use_dynamic_chunk,
    use_dynamic_left_chunk,
    decoding_chunk_size,
    static_chunk_size,
    num_decoding_left_chunks,
):
    """Apply optional mask for encoder.

    Args:
        xs_len (int): padded input, 1/4 ori data length
        mask (numpy.array): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks

    Returns:
        numpy.array: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    masks = masks.astype(np.bool)
    if use_dynamic_chunk:
        max_len = xs_len
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = np.random.randint(1, max_len, (1,)).tolist()[0]
            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = np.random.randint(
                        0, max_left_chunks, (1,)
                    ).tolist()[0]
        chunk_masks = subsequent_chunk_mask(
            xs_len, chunk_size, num_left_chunks
        )  # (L, L)
        chunk_masks = np.expand_dims(chunk_masks, 0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(
            xs_len, static_chunk_size, num_left_chunks
        )  # (L, L)
        chunk_masks = np.expand_dims(chunk_masks, 0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks
