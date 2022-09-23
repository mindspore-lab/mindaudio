from typing import List
import numpy as np


TRAIN_INPUT_PAD_LENGTH = 1250
TRAIN_LABEL_PAD_LENGTH = 480
TEST_INPUT_PAD_LENGTH = 3500
BLANK_ID = 0 # index of '_' in tokenisers


def pad_batch_text_tokens_train(batch_script: List, max_tgt_len: int = None):
    batch_size = len(batch_script)
    target_indices = []
    if max_tgt_len is None:
        max_tgt_len = max(len(script) for script in batch_script)
    targets = np.ones((batch_size, max_tgt_len), dtype=np.int32) * BLANK_ID
    for k, scripts_ in zip(range(batch_size), batch_script):
        script_length = len(scripts_)
        targets[k, :script_length] = scripts_
        for m in range(max_tgt_len):
            target_indices.append([k, m])
    targets = np.reshape(targets, (-1,))
    return np.array(target_indices, dtype=np.int64), np.array(targets, dtype=np.int32)


def pad_batch_text_tokens_eval(batch_script: List):
    batch_size = len(batch_script)
    target_indices = []
    targets = []
    for k, scripts_ in zip(range(batch_size), batch_script):
        script_length = len(scripts_)
        targets.extend(scripts_)
        for m in range(script_length):
            target_indices.append([k, m])
    return np.array(target_indices, dtype=np.int64), np.array(targets, dtype=np.int32)


def pad_trim_batch_feature_train(batch_feat: List):
    batch_size = len(batch_feat)
    freq_size = np.shape(batch_feat[-1])[0]
    inputs = np.zeros((batch_size, 1, freq_size, TRAIN_INPUT_PAD_LENGTH), dtype=np.float32)
    input_length = np.zeros(batch_size, np.float32)
    for k, feat in zip(range(batch_size), batch_feat):
        seq_length = np.shape(feat)[1]
        if seq_length <= TRAIN_INPUT_PAD_LENGTH:
            input_length[k] = seq_length
            inputs[k, 0, :, :seq_length] = feat[:, :seq_length]
        else:
            maxstart = seq_length - TRAIN_INPUT_PAD_LENGTH
            start = np.random.randint(maxstart)
            input_length[k] = TRAIN_INPUT_PAD_LENGTH
            inputs[k, 0, :, :TRAIN_INPUT_PAD_LENGTH] = feat[:, start:start + TRAIN_INPUT_PAD_LENGTH]
    return inputs, input_length


def pad_trim_batch_feature_eval(batch_feat: List):
    batch_size = len(batch_feat)
    freq_size = np.shape(batch_feat[-1])[0]
    inputs = np.zeros((batch_size, 1, freq_size, TEST_INPUT_PAD_LENGTH), dtype=np.float32)
    input_length = np.zeros(batch_size, np.float32)
    for k, feat in zip(range(batch_size), batch_feat):
        seq_length = np.shape(feat)[1]
        input_length[k] = seq_length
        inputs[k, 0, :, :seq_length] = feat
    return inputs, input_length


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