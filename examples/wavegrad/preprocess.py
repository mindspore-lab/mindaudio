import sys
from multiprocessing import Pool, cpu_count

import mindspore as ms
import numpy as np
from dataset import FEATURE_POSTFIX, WAV_POSTFIX
from ljspeech import LJSpeech, create_ljspeech_tts_dataset
from mindspore.dataset.audio import MelScale, Spectrogram
from tqdm import tqdm

from mindaudio.data.io import read

sys.path.append("..")
from config import load_config


def read_wav(filename):
    filename = str(filename).replace("b'", "").replace("'", "")
    audio, _ = read(filename)
    signed_int16_max = 2**15
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / signed_int16_max
    audio = audio / np.max(np.abs(audio))
    return audio, audio, filename


def _normalize(S):
    S = 20 * np.log10(np.clip(S, 1e-5, None)) - 20
    S = np.clip((S + 100) / 100, 0.0, 1.0)
    return S.astype(np.float32)


def create_prep_dataset(hps, is_train):
    ds = LJSpeech(
        data_path=hps.data_path, manifest_path=hps.manifest_path, is_train=is_train
    )
    ds = create_ljspeech_tts_dataset(ds, rank=0, group_size=1)

    # process audio: file -> wav -> feature
    ds = ds.map(
        input_columns=["audio"],
        output_columns=["audio", "mel", "filename"],
        column_order=["audio", "mel", "filename"],
        operations=read_wav,
        num_parallel_workers=cpu_count(),
    )

    stft = Spectrogram(
        n_fft=hps.n_fft,
        win_length=hps.hop_samples * 4,
        hop_length=hps.hop_samples,
        power=1.0,
        center=True,
    )

    mel = MelScale(
        n_mels=hps.n_mels,
        sample_rate=hps.sample_rate,
        f_min=20.0,
        f_max=hps.sample_rate / 2.0,
        n_stft=hps.n_fft // 2 + 1,
    )

    ds = ds.map(
        input_columns=["mel"],
        column_order=["audio", "mel", "filename"],
        operations=[stft, mel, _normalize],
        num_parallel_workers=cpu_count(),
    )

    return ds


def preprocess_ljspeech(hps, is_train):
    ds = create_prep_dataset(hps, is_train)
    it = ds.create_dict_iterator()

    results = []
    pool = Pool(processes=cpu_count())

    for x in tqdm(it, total=ds.get_dataset_size()):
        npy = x["audio"].asnumpy()
        filename = str(x["filename"]).replace(".wav", WAV_POSTFIX)
        results.append(pool.apply_async(func=np.save, args=[filename, npy]))

        npy = x["mel"].asnumpy()
        filename = str(x["filename"]).replace(".wav", FEATURE_POSTFIX)
        results.append(pool.apply_async(func=np.save, args=[filename, npy]))
    for r in tqdm(results):
        r.get()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device_target",
        "-d",
        type=str,
        default="CPU",
        choices=("GPU", "CPU", "Ascend"),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="wavegrad_base.yaml",
    )
    parser.add_argument("--device_id", "-i", type=int, default=0)
    args = parser.parse_args()
    ms.context.set_context(
        mode=ms.context.PYNATIVE_MODE,
        device_target=args.device_target,
        device_id=args.device_id,
    )
    hps = load_config(args.config)
    preprocess_ljspeech(hps=hps, is_train=True)
    preprocess_ljspeech(hps=hps, is_train=False)
