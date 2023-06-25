import mindspore as ms
import numpy as np


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        _, max_len = ms.ops.max(lengths)
    ids = ms.numpy.arange(0, int(max_len)).expand_dims(0)
    mask = ids >= lengths.expand_dims(1)

    return mask


def get_mask_from_lengths_np(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    ids = np.arange(0, max_len)[None, ...]
    mask = ids >= lengths[:, None, ...]

    return mask


def pad(input_ele, mel_max_length=None):
    # [b t c]
    axis = 0
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].shape[axis] for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = ms.ops.pad(batch, ((0, max_len - batch.shape[axis]),))
            out_list.append(one_batch_padded)
        elif len(batch.shape) == 2:
            one_batch_padded = ms.ops.pad(
                batch, ((0, max_len - batch.shape[axis]), (0, 0))
            )
            out_list.append(one_batch_padded)
    out_padded = ms.ops.stack(out_list)
    return out_padded
