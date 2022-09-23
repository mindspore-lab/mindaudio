"""
Create train or eval dataset.
"""
import sys
import numpy as np
from multiprocessing import cpu_count

sys.path.append('..')
from examples.tokenisers import CharTokeniser
from mindaudio.data.collate import (
    pad_batch_text_tokens_train,
    pad_trim_batch_feature_train,
    pad_trim_batch_feature_eval,
    pad_batch_text_tokens_eval,
)
from mindaudio.data.datasets import LibrittsASR, create_asr_dataset


SAVE_POSTFIX = '_deepspeech2_feat.npy'


def create_dataset(
    data_path,
    manifest_path,
    labels,
    normalize=True,
    train_mode=True,
    batch_size=1,
    rank=0,
    group_size=1,
):
    # raw dataset, each item: [path_to_wav, path_to_text]

    ds = LibrittsASR(
        data_path=data_path,
        manifest_path=manifest_path,
    )
    ds = create_asr_dataset(ds, rank, group_size)

    # process text: file -> text -> token ids

    tokeniser = CharTokeniser(labels=labels)
    def read_text(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        text = open(filename, encoding='utf8').readline().replace('\n', '')
        ids = tokeniser.tokenise(text)
        return ids, ids.shape[0]

    ds = ds.map(
        input_columns=['text'],
        operations=read_text,
        output_columns=['targets', 'targets_length'],
        column_order=['audio', 'targets', 'targets_length'],
        num_parallel_workers=cpu_count(),
    )

    # process audio: file -> wav -> feature

    def read_feat(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        x = np.load(filename.replace('.wav', SAVE_POSTFIX))
        x = np.log1p(x)
        if normalize:
            x = (x - x.mean()) / x.std()
        return x

    ds = ds.map(
        input_columns=['audio'],
        operations=read_feat,
        num_parallel_workers=cpu_count(),
    )

    # pad to same length, then make a batch
    def batch_collate_train(audio, targets, targets_length, unused_batch_info=None):
        inputs, input_length = pad_trim_batch_feature_train(audio)
        target_indices, targets = pad_batch_text_tokens_train(targets)
        return inputs, input_length, target_indices, targets

    def batch_collate_eval(audio, targets, targets_length, unused_batch_info=None):
        inputs, input_length = pad_trim_batch_feature_eval(audio)
        target_indices, targets = pad_batch_text_tokens_eval(targets)
        return inputs, input_length, target_indices, targets

    ds = ds.batch(
        batch_size, 
        per_batch_map=batch_collate_train if train_mode else batch_collate_eval,
        input_columns=['audio', 'targets', 'targets_length'],
        output_columns=['inputs', 'input_length', 'target_indices', 'label_values'],
        column_order=['inputs', 'input_length', 'target_indices', 'label_values'],
    )

    return tokeniser.vocab_size, ds
