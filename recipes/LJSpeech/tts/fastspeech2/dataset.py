import os
import numpy as np
from multiprocessing import cpu_count
import mindspore as ms

from recipes.LJSpeech import LJSpeech
from recipes.LJSpeech.tts import create_ljspeech_tts_dataset
from mindaudio.models.transformer import get_sinusoid_encoding_table


TEXT_DIR = 'fs2_phoneme'
WAV_DIR = 'fs2_wav'
MEL_DIR = 'fs2_mel'
ENERGY_DIR = 'fs2_energy'
PITCH_DIR = 'fs2_pitch'
DURATION_DIR = 'fs2_duration'

TEXT_POSTFIX = '_phoneme.npy'
WAV_POSTFIX = '_wav.npy'
MEL_POSTFIX = '_mel.npy'
ENERGY_POSTFIX = '_energy.npy'
PITCH_POSTFIX = '_pitch.npy'
DURATION_POSTFIX = '_duration.npy'

feature_columns = ['phoneme', 'wav', 'mel', 'energy', 'pitch', 'duration']
all_dirs = dict(zip(feature_columns, [
    TEXT_DIR,
    WAV_DIR,
    MEL_DIR,
    ENERGY_DIR,
    PITCH_DIR,
    DURATION_DIR,
]))
all_postfix = dict(zip(feature_columns, [
    TEXT_POSTFIX,
    WAV_POSTFIX,
    MEL_POSTFIX,
    ENERGY_POSTFIX,
    PITCH_POSTFIX,
    DURATION_POSTFIX,
]))

len_max_seq = 1000
d_word_vec = 256
positional_embeddings = get_sinusoid_encoding_table(len_max_seq + 1, d_word_vec, padding_idx=None)

data_columns = [
    'speakers',
    'texts',
    'src_lens',
    'max_src_len',
    'positions_encoder',
    'positions_decoder',
    'mels',
    'mel_lens',
    'max_mel_len',
    'p_targets',
    'e_targets',
    'd_targets',
]


def create_dataset(data_path, manifest_path, batch_size, is_train=True, rank=0, group_size=1):
    ds = LJSpeech(
        data_path=data_path,
        manifest_path=manifest_path,
        is_train=is_train,
    )
    ds = create_ljspeech_tts_dataset(ds, rank=rank, group_size=group_size)

    def read_feat(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        base = filename[filename.rfind('/')+1: ].replace('.wav', '')

        data = []
        for key in feature_columns:
            data.append(np.load(os.path.join(data_path, all_dirs[key], base + all_postfix[key])))
            if key == 'mel':
                data[-1] = data[-1].T

        return tuple(data)

    ds = ds.map(
        input_columns=['audio'],
        output_columns=feature_columns,
        column_order=feature_columns,
        operations=[read_feat],
        num_parallel_workers=cpu_count(),
    )

    def pad_to_max(xs):
        B = len(xs)
        T = max(x.shape[0] for x in xs)
        shape = [B, T] + list(xs[0].shape[1:])
        ys = np.zeros(shape, dtype=np.float32)
        lengths = np.zeros(B, np.int32)
        for i, x in enumerate(xs):
            ys[i, : x.shape[0]] = x
            lengths[i] = x.shape[0]
        return ys, lengths, np.array(T, np.int32)

    def batch_collate(phonemes, wavs, mels, pitch, energy, duration, unused_batch_info=None):
        phonemes, src_lens, max_src_len = pad_to_max(phonemes)
        mels, mel_lens, max_mel_len = pad_to_max(mels)
        pitch, _, _ = pad_to_max(pitch)
        energy, _, _ = pad_to_max(energy)
        duration, _, _ = pad_to_max(duration)
        speakers = np.zeros(len(phonemes), np.float32)
        positions_encoder = positional_embeddings[None, : max_src_len].repeat(len(phonemes), 0)
        max_duration = duration.sum(-1).max().astype(np.int32)
        positions_decoder = positional_embeddings[None, : max_duration].repeat(len(phonemes), 0)
        return (
            speakers,
            phonemes,
            src_lens,
            max_src_len,
            positions_encoder,
            positions_decoder,
            mels,
            mel_lens,
            max_mel_len,
            pitch,
            energy,
            duration.astype(np.int32),
        )
    ds = ds.batch(
        batch_size, 
        per_batch_map=batch_collate,
        input_columns=feature_columns,
        output_columns=data_columns,
        column_order=data_columns,
        drop_remainder=True,
        python_multiprocessing=False,
        num_parallel_workers=cpu_count()
    )

    return ds
