"""
Create train or eval dataset.
"""
import sys
import numpy as np
from multiprocessing import cpu_count

sys.path.append('..')
from mindaudio.data.datasets import LJSpeechTTS, create_asr_dataset
from examples.tokenisers import ARPAbetTokeniser
from mindaudio.data.collate import pad_sequence


SAVE_POSTFIX = '_tacotron2_feat.npy'


__all__ = [
    'SAVE_POSTFIX',
    'create_dataset'
]


def create_dataset(
    manifest_path,
    hparams,
    batch_size,
    rank=0,
    group_size=1,
):

    # raw dataset, each item: [path_to_wav, path_to_text]

    ds = LJSpeechTTS(
        manifest_path=manifest_path,
    )
    ds = create_asr_dataset(ds, rank, group_size)

    # process text: file -> text -> token ids

    tokeniser = ARPAbetTokeniser()
    def read_text(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        text = open(filename, encoding='utf8').readline().replace('\n', '') + '~'
        ids = tokeniser.tokenise(text)
        return ids

    ds = ds.map(
        input_columns=['text'],
        operations=read_text,
        num_parallel_workers=cpu_count(),
    )

    # process audio: file -> wav -> feature

    def read_feat(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        x = np.load(filename.replace('.wav', SAVE_POSTFIX))
        return x

    ds = ds.map(
        input_columns=['audio'],
        operations=read_feat,
        num_parallel_workers=cpu_count(),
    )

    # pad xs and ys to same length respectively, then make a batch
    def batch_collate(text, audio, unused_batch_info=None):
        xs, ys = [], []
        max_txt_len, max_mel_len = 0, 0
        for x, y in zip(text, audio):
            xs.append(x)
            ys.append(y.T)
            max_txt_len = max(max_txt_len, x.shape[0])
            max_mel_len = max(max_mel_len, y.shape[1])
        max_mel_len += hparams.n_frames_per_step - max_mel_len % hparams.n_frames_per_step

        xs_lengths = np.array([x.shape[0] for x in xs], dtype=np.int32)

        xs_pad = pad_sequence(
            xs,
            padding_value=1,
            padding_max_len=max_txt_len,
            atype=np.int32,
        )
        ys_pad = pad_sequence(
            ys,
            padding_value=0.,
            padding_max_len=max_mel_len,
            atype=np.float32,
        ).transpose([0, 2, 1]) # [b, c, t]

        B = xs_pad.shape[0]
        mel_mask = np.zeros((B, max_mel_len)).astype(bool)
        gate_padded = np.zeros((B, max_mel_len), np.float32)
        for i, y in enumerate(ys):
            mel_mask[i, : y.shape[0]] = True
            gate_padded[i, y.shape[0] - 1:] = 1

        # [B, n_mels, max_mel_len]
        mel_mask = np.expand_dims(mel_mask, 1).repeat(y.shape[1], 1)

        rnn_mask = np.zeros((B, max_txt_len)).astype(bool)
        text_mask = np.zeros((B, max_txt_len)).astype(bool)
        for i, x in enumerate(xs):
            text_mask[i, :x.shape[0]] = True
            rnn_mask[i, :x.shape[0]] = True

        rnn_mask = np.expand_dims(rnn_mask, 2).repeat(512, 2)

        return (
            xs_pad,
            xs_lengths,
            ys_pad,
            gate_padded,
            text_mask,
            mel_mask,
            rnn_mask
        )

    output_feat = [
        'text_padded', 
        'input_lengths', 
        'mel_padded', 
        'gate_padded', 
        'text_mask', 
        'mel_mask', 
        'rnn_mask'
    ]
    ds = ds.batch(
        batch_size, 
        per_batch_map=batch_collate,
        input_columns=['text'] + ['audio'],
        output_columns=output_feat,
        column_order=output_feat,
        python_multiprocessing=True
    )

    return tokeniser.vocab_size, ds
