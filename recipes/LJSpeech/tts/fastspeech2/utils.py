import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from math import log as ln


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table.astype(np.float32)


def get_mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        _, max_len = ms.ops.max(lengths)
    ids = ms.numpy.arange(0, int(max_len)).expand_dims(0)
    # ids = (ms.ops.ones(int(max_len), ms.float32).cumsum() - 1).expand_dims(0)
    mask = ids >= lengths.expand_dims(1)

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
            one_batch_padded = ms.ops.pad(
                batch, ((0, max_len - batch.shape[axis]),)
            )
            out_list.append(one_batch_padded)
        elif len(batch.shape) == 2:
            one_batch_padded = ms.ops.pad(
                batch, ((0, max_len - batch.shape[axis]), (0, 0))
            )
            out_list.append(one_batch_padded)
    out_padded = ms.ops.stack(out_list)
    return out_padded
