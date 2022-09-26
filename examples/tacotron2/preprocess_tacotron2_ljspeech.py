# Given the path to ljspeech/wavs,
# this script converts wav files to .npy features used for training.
# python3 preprocess_tacotron2_ljspeech.py

import sys
import numpy as np
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from mindspore.dataset.audio import Spectrogram, MelScale, AmplitudeToDB

sys.path.append('.')
from examples.tacotron2.dataset import SAVE_POSTFIX
from examples.tacotron2.hparams import hparams as hps
from mindaudio.data.io import read
from mindaudio.data.datasets import LJSpeechTTS, create_asr_dataset
from mindaudio.data.features import trim
from mindaudio.models.tacotron2.config import config


def _normalize(S):
    ''' normalize '''
    return np.clip((S - hps.min_level_db) / -hps.min_level_db, 0, 1)


def create_prep_dataset(hparams, data_path, manifest_path):
    ds = LJSpeechTTS(
        data_path=data_path,
        manifest_path=manifest_path,
    )
    ds = create_asr_dataset(ds, rank=0, group_size=1)

    # process audio: file -> wav -> feature

    def read_wav(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        audio, _ = read(filename)
        signed_int16_max = 2**15
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / signed_int16_max
        audio = audio / np.max(np.abs(audio))
        return audio, filename

    ds = ds.map(
        input_columns=['audio'],
        output_columns=['audio', 'filename'],
        column_order=['audio', 'filename'],
        operations=read_wav,
        num_parallel_workers=cpu_count(),
    )

    def trim_with_args(x):
        hop_length = 512
        x, _ = trim(x, top_db=60, reference=np.max, frame_length=2048, hop_length=hop_length)
        x = np.concatenate((x, np.zeros(5 * hop_length, dtype=np.float32)), 0)
        return x

    stft = Spectrogram(
        n_fft=(hparams.num_freq - 1) * 2,
        win_length=hparams.win_length,
        hop_length=hparams.hop_length,
        power=hparams.power,
        center=True,
    )

    mel = MelScale(
        n_mels=hparams.num_mels,
        sample_rate=hparams.sample_rate,
        f_min=hparams.fmin, 
        f_max=hparams.fmax,
        n_stft=hparams.num_freq,
    )

    a2d = AmplitudeToDB()

    ds = ds.map(
        input_columns=['audio'],
        column_order=['audio', 'filename'],
        operations=[trim_with_args, stft, mel, a2d, _normalize],
        num_parallel_workers=cpu_count(),
    )

    return ds


def preprocess_tacotron2_ljspeech(hparams, data_path, manifest_path):
    ds = create_prep_dataset(hparams, data_path, manifest_path)
    it = ds.create_dict_iterator()

    results = []
    pool = Pool(processes=cpu_count())

    for x in tqdm(it, total=ds.get_dataset_size()):
        npy = x['audio'].asnumpy()
        filename = str(x['filename']).replace('.wav', SAVE_POSTFIX)
        results.append(
            pool.apply_async(func=np.save, args=[filename, npy])
        )
    for r in tqdm(results):
        r.get()


if __name__ == '__main__':
    preprocess_tacotron2_ljspeech(
        hparams=hps, 
        data_path=config.data_path,
        manifest_path=config.manifest_path,
    )
