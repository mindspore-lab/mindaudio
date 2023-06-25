import os
from multiprocessing import cpu_count

import numpy as np

from mindaudio.models.transformer import get_sinusoid_encoding_table
from mindaudio.models.fastspeech2.utils import get_mask_from_lengths_np
from ljspeech import LJSpeech, create_ljspeech_tts_dataset
import sys
sys.path.append('..')
from config import load_config
hps = load_config('fastspeech2.yaml')

TEXT_DIR = "fs2_phoneme"
WAV_DIR = "fs2_wav"
MEL_DIR = "fs2_mel"
ENERGY_DIR = "fs2_energy"
PITCH_DIR = "fs2_pitch"
DURATION_DIR = "fs2_duration"

TEXT_POSTFIX = "_phoneme.npy"
WAV_POSTFIX = "_wav.npy"
MEL_POSTFIX = "_mel.npy"
ENERGY_POSTFIX = "_energy.npy"
PITCH_POSTFIX = "_pitch.npy"
DURATION_POSTFIX = "_duration.npy"

feature_columns = ["phoneme", "wav", "mel", "energy", "pitch", "duration"]
all_dirs = dict(
    zip(
        feature_columns,
        [
            TEXT_DIR,
            WAV_DIR,
            MEL_DIR,
            ENERGY_DIR,
            PITCH_DIR,
            DURATION_DIR,
        ],
    )
)
all_postfix = dict(
    zip(
        feature_columns,
        [
            TEXT_POSTFIX,
            WAV_POSTFIX,
            MEL_POSTFIX,
            ENERGY_POSTFIX,
            PITCH_POSTFIX,
            DURATION_POSTFIX,
        ],
    )
)

len_max_seq = 1000
d_word_vec = 256
positional_embeddings = get_sinusoid_encoding_table(
    len_max_seq + 1, d_word_vec, padding_idx=None
)

data_columns = [
    "speakers",
    "texts",
    "src_lens",
    "max_src_len",
    "positions_encoder",
    "positions_decoder",
    "mels",
    "mel_lens",
    "max_mel_len",
    "p_targets",
    "e_targets",
    "d_targets",
]


def create_dataset(
    data_path, manifest_path, batch_size, is_train=True, rank=0, group_size=1
):
    ds = LJSpeech(
        data_path=data_path,
        manifest_path=manifest_path,
        is_train=is_train,
    )
    ds = create_ljspeech_tts_dataset(ds, rank=rank, group_size=group_size)

    def read_feat(filename):
        filename = str(filename).replace("b'", "").replace("'", "")
        base = filename[filename.rfind("/") + 1 :].replace(".wav", "")

        data = []
        for key in feature_columns:
            x = np.load(os.path.join(data_path, all_dirs[key], base + all_postfix[key]))
            if key == 'mel':
                x = x.T
            if key not in ['phoneme', 'duration']:
                x = x.astype(np.float32)
            else:
                x = x.astype(np.int32)
            data.append(x)
        return tuple(data)

    ds = ds.map(
        input_columns=['audio'],
        output_columns=feature_columns,
        column_order=feature_columns,
        operations=[read_feat],
        num_parallel_workers=cpu_count(),
    )

    def pad_to_max(xs, T=None):
        B = len(xs)
        if T is None:
            T = max(x.shape[0] for x in xs)
        shape = [B, T] + list(xs[0].shape[1:])
        ys = np.zeros(shape, dtype=xs[0].dtype)
        lengths = np.zeros(B, np.int32)
        for i, x in enumerate(xs):
            ys[i, : x.shape[0]] = x
            lengths[i] = x.shape[0]
        return ys, lengths, np.array(T, np.int32)

    def batch_collate(phonemes, wavs, mels, pitch, energy, duration, unused_batch_info=None):
        B = len(phonemes)
        dtype = np.float16 if hps.use_fp16 else np.float32
        dataset_max_src_len = 135
        dataset_max_mel_len = 742
        phonemes, src_lens, max_src_len = pad_to_max(phonemes, T=dataset_max_src_len)
        mels, mel_lens, max_mel_len = pad_to_max(mels, T=dataset_max_mel_len)
        pitch, _, _ = pad_to_max(pitch, T=dataset_max_mel_len)
        energy, _, _ = pad_to_max(energy, T=dataset_max_mel_len)
        duration, _, _ = pad_to_max(duration, T=dataset_max_src_len)
        speakers = np.zeros(B, dtype)
        positions_encoder = positional_embeddings[None, : max_src_len].repeat(B, 0)
        max_duration = duration.sum(-1).max().astype(np.int32)
        max_duration = max_mel_len
        positions_decoder = positional_embeddings[None, : max_duration].repeat(B, 0)
        expanded_phonemes = []
        for p, d, y in zip(phonemes, duration, mels):
            xh = np.repeat(p, d)
            x = np.zeros(y.shape[0], dtype=xh.dtype)
            x[: xh.shape[0]] = xh
            expanded_phonemes.append(x)
        expanded_phonemes, expanded_src_lens, expanded_max_src_len = pad_to_max(expanded_phonemes, T=dataset_max_mel_len)
        src_masks = get_mask_from_lengths_np(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths_np(mel_lens, max_mel_len) if mel_lens is not None else None
        return (
            speakers.astype(dtype),
            phonemes.astype(np.int32),
            src_lens,
            max_src_len,
            positions_encoder.astype(dtype),
            positions_decoder.astype(dtype),
            mels.astype(dtype),
            mel_lens,
            max_mel_len,
            pitch.astype(dtype),
            energy.astype(dtype),
            duration.astype(np.int32),
            expanded_phonemes.astype(np.int32),
            expanded_src_lens,
            expanded_max_src_len,
            src_masks,
            mel_masks,
        )
    global data_columns
    dc = data_columns + [
        'expanded_phonemes',
        'expanded_src_lens',
        'expanded_max_src_len',
        'src_masks',
        'mel_masks',
    ]
    ds = ds.batch(
        batch_size, 
        per_batch_map=batch_collate,
        input_columns=feature_columns,
        output_columns=dc,
        column_order=dc,
        drop_remainder=True,
        python_multiprocessing=False,
        num_parallel_workers=cpu_count()
    )

    return ds
