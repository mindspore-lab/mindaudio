import argparse
import ast
import os

import mindspore as ms
import numpy as np
from mindspore.dataset.audio import MelScale, Spectrogram
from tqdm import tqdm

import mindaudio
from mindaudio.data.io import write
from recipes.LJSpeech.tts.wavegrad.preprocess import _normalize, read_wav


def parse_args():
    parser = argparse.ArgumentParser(description="WaveGrad training")
    parser.add_argument(
        "--device_target", type=str, default="CPU", choices=("GPU", "CPU", "Ascend")
    )
    parser.add_argument("--device_id", "-i", type=int, default=0)
    parser.add_argument("--save", "-s", type=str, default="results")
    parser.add_argument("--plot", "-p", type=ast.literal_eval, default=True)
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="recipes/LJSpeech/tts/wavegrad/wavegrad_base.yaml",
    )
    parser.add_argument("--restore", "-r", type=str, default="")
    parser.add_argument("--restore_url", "-u", type=str, default="")
    parser.add_argument("--wav", "-w", type=str, default="data/LJspeech-1.1/wavs")
    parser.add_argument("--mel", "-m", type=str, default=None)
    parser.add_argument("--data_url", default="")
    parser.add_argument("--train_url", default="")
    args = parser.parse_args()
    return args


args = parse_args()
hps = mindaudio.load_config(args.config)
os.makedirs(args.save, exist_ok=True)
ms.context.set_context(
    mode=ms.context.PYNATIVE_MODE,
    device_target=args.device_target,
    device_id=args.device_id,
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

if args.mel is not None:
    wav = None
    feature = np.load(args.mel)
    if len(feature.shape) == 2:
        feature = feature[None, ...]
    if feature.shape[-1] == hps.n_mels:
        feature = feature.transpose([0, 2, 1])
    old = args.mel.split("/")[-1].replace(".npy", "")
else:
    wav = read_wav(args.wav)[0]
    feature = stft(wav)
    feature = mel(feature)
    feature = _normalize(feature)[None, ...]
    old = args.wav.split("/")[-1].replace(".wav", "")

gtwav = wav
gtmel = feature[0]

feature = ms.Tensor(feature)
print("old:", old)
print("feature:", feature.shape)

model, ckpt = mindaudio.create_model("WaveGrad", hps, args.restore, is_train=False)
global_step = 0
if ckpt is not None:
    if "cur_step" in ckpt:
        global_step = int(ckpt["cur_step"].asnumpy())
print("restore:", global_step)

hps.noise_schedule = np.linspace(
    hps.noise_schedule_start, hps.noise_schedule_end, hps.noise_schedule_S
)
beta = hps.noise_schedule
alpha = 1 - beta
alpha_cum = np.cumprod(alpha).astype(np.float32)

audio = np.random.normal(
    0, 1, [feature.shape[0], hps.hop_samples * feature.shape[-1]]
).astype(np.float32)
print("audio:", audio.shape)
noise_scale = ms.Tensor(alpha_cum**0.5)[:, None]

S = len(alpha)
mels = []
wavs = []
for m in tqdm(range(S)):
    n = S - 1 - m
    c1 = 1 / alpha[n] ** 0.5
    c2 = (1 - alpha[n]) / (1 - alpha_cum[n]) ** 0.5
    pred = model(ms.Tensor(audio), noise_scale[n], feature).asnumpy()
    audio = c1 * (audio - c2 * pred)
    if n > 0:
        noise = np.random.normal(0, 1, audio.shape)
        sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
        audio += sigma * noise

    if (m + 1) > (S - 50) and (m + 1) % 5 == 0:
        write(
            args.save + "/%d_predicted_%s_%d.wav" % (global_step, old, m + 1),
            audio[0],
            hps.sample_rate,
        )
        wavs.append(audio[0])


# BELOW: plots gif for last steps of reverse diffusion process
if not args.plot:
    exit()
else:
    import matplotlib.pyplot as plt  # pylint: disable=E402
    from librosa import display  # pylint: disable=E402
    from matplotlib.animation import FuncAnimation  # pylint: disable=E402

for wav in wavs:
    feat = _normalize(mel(stft(wav)))
    mels.append(feat)

fig = plt.figure(figsize=(24, 8))
plt.margins(x=0, y=0, tight=True)

titles = ["original %s" % old, "generated %s" % old]
n_rows = 2
subfigs = fig.subfigures(nrows=n_rows, ncols=1)
axes = []
for i, subfig in enumerate(subfigs):
    ax = subfigs[i].subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    subfigs[i].suptitle(titles[i])
    axes.append(ax)

if gtwav is not None:
    display.waveshow(gtwav, sr=hps.sample_rate, ax=axes[0][0])
axes[0][1].imshow(gtmel, origin="lower")


def plot(wav, mel, frame):
    display.waveshow(wav, sr=hps.sample_rate, ax=axes[1][0], label="%s" % frame)
    axes[1][0].legend()
    img = axes[1][1].imshow(mel, origin="lower")
    return (img,)


def init():
    return plot(wavs[0], mels[0], S - 50 + 5)


def update(frame):
    axes[1][0].clear()
    axes[1][1].clear()
    return plot(wavs[frame], mels[frame], S - 50 + 5 * (frame + 1))


ani = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=len(wavs),
    interval=1,
    repeat_delay=5,
    # blit=True,
)

ani.save(args.save + "/%d_%s.gif" % (global_step, old), fps=2, writer="imagemagick")
