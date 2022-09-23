# Given the path to Librispeech/train-clean-100,
# this script converts wav files to .npy features used for training.
# python3 preprocess_deepspeech2_librispeech.py 


import sys
import numpy as np
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from mindspore.dataset.audio import Magphase

sys.path.append('.')
from examples.deepspeech2.dataset import SAVE_POSTFIX
from examples.deepspeech2.config import train_config
from mindaudio.data.io import read
from mindaudio.data.datasets import LibrittsASR, create_asr_dataset
from mindaudio.data.features import stft


def create_prep_dataset(
    audio_conf,
    data_path,
    manifest_path
):
    # raw dataset, each item: [path_to_wav, path_to_text]

    ds = LibrittsASR(
        data_path=data_path,
        manifest_path=manifest_path,
    )
    ds = create_asr_dataset(ds, rank=0, group_size=1)

    # process audio: file -> wav -> feature

    def read_wav(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        audio, _ = read(filename)
        audio = (audio + 32768) / (32767 + 32768) * 2 - 1
        return audio, filename

    ds = ds.map(
        input_columns=['audio'],
        output_columns=['audio', 'filename'],
        column_order=['audio', 'filename'],
        operations=read_wav,
        num_parallel_workers=cpu_count(),
    )

    def stft_with_args(x):
        return stft(
            waveforms=x,
            n_fft=int(audio_conf.sample_rate * audio_conf.window_size),
            win_length=int(audio_conf.sample_rate * audio_conf.window_size),
            hop_length=int(audio_conf.sample_rate * audio_conf.window_stride),
            window=audio_conf.window,
            center=True,
            pad_mode="reflect",
            return_complex=False
        )

    _magphase = Magphase(power=1.0)
    def magphase_with_args(x):
        y = _magphase(x)[0]
        return y

    ds = ds.map(
        input_columns=['audio'],
        column_order=['audio', 'filename'],
        operations=[stft_with_args, magphase_with_args],
        num_parallel_workers=cpu_count(),
    )

    return ds


def preprocess_deepspeech2_librispeech(audio_conf, data_path, manifest_path):
    ds = create_prep_dataset(audio_conf, data_path, manifest_path)
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
    preprocess_deepspeech2_librispeech(
        audio_conf=train_config.DataConfig.SpectConfig,
        data_path=train_config.DataConfig.data_path,
        manifest_path=train_config.DataConfig.train_manifest,
    )
