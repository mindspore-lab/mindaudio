from multiprocessing import cpu_count

import numpy as np
from ljspeech import LJSpeech, create_ljspeech_tts_dataset

FEATURE_POSTFIX = "_feature.npy"
WAV_POSTFIX = "_wav.npy"


def diffuse(x, S, noise_level):
    s = np.random.randint(1, S + 1)
    l_a, l_b = noise_level[s - 1], noise_level[s]
    r = np.random.rand()
    noise_scale = (l_a + r * (l_b - l_a)).astype(np.float32)
    noise = np.random.randn(*(x.shape)).astype(np.float32)
    noisy_audio = noise_scale * x + (1.0 - noise_scale**2) ** 0.5 * noise

    return noisy_audio.astype(np.float32), noise_scale, noise


def create_wavegrad_dataset(hps, batch_size, is_train=True, rank=0, group_size=1):
    ds = LJSpeech(
        data_path=hps.data_path,
        manifest_path=hps.manifest_path,
        is_train=is_train,
    )
    ds = create_ljspeech_tts_dataset(ds, rank=rank, group_size=group_size)

    def read_feat(filename):
        filename = str(filename).replace("b'", "").replace("'", "")
        x = np.load(filename.replace(".wav", WAV_POSTFIX))
        c = np.load(filename.replace(".wav", FEATURE_POSTFIX))
        return x, c

    input_columns = ["audio", "spectrogram"]
    ds = ds.map(
        input_columns=["audio"],
        output_columns=input_columns,
        column_order=input_columns,
        operations=[read_feat],
        num_parallel_workers=cpu_count(),
    )

    hps.noise_schedule = np.linspace(
        hps.noise_schedule_start, hps.noise_schedule_end, hps.noise_schedule_S
    )
    beta = hps.noise_schedule
    noise_level = np.cumprod(1 - beta) ** 0.5
    noise_level = np.concatenate([[1.0], noise_level], axis=0).astype(np.float32)

    output_columns = ["noisy_audio", "noise_scale", "noise", "spectrogram"]

    def batch_collate(audio, spectrogram, unused_batch_info=None):
        batch_noisy_audio, batch_noise_scale, batch_noise, batch_spectrogram = (
            [],
            [],
            [],
            [],
        )
        samples_per_frame = hps.hop_samples
        for x, c in zip(audio, spectrogram):
            start = np.random.randint(0, c.shape[1] - hps.crop_mel_frames)
            end = start + hps.crop_mel_frames
            c = c[:, start:end]
            batch_spectrogram.append(c)

            start *= samples_per_frame
            end *= samples_per_frame
            x = x[start:end]
            x = np.pad(x, (0, (end - start) - len(x)), mode="constant")

            noisy_audio, noise_scale, noise = diffuse(
                x, hps.noise_schedule_S, noise_level
            )

            batch_noisy_audio.append(noisy_audio)
            batch_noise_scale.append(noise_scale)
            batch_noise.append(noise)

        batch_noisy_audio = np.stack(batch_noisy_audio)
        batch_noise_scale = np.stack(batch_noise_scale)
        batch_noise = np.stack(batch_noise)
        batch_spectrogram = np.stack(batch_spectrogram)

        return batch_noisy_audio, batch_noise_scale, batch_noise, batch_spectrogram

    ds = ds.batch(
        batch_size,
        per_batch_map=batch_collate,
        input_columns=input_columns,
        output_columns=output_columns,
        column_order=output_columns,
        drop_remainder=True,
        python_multiprocessing=False,
        num_parallel_workers=cpu_count(),
    )

    return ds
