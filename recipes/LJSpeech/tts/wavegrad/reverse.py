import numpy as np
from tqdm import tqdm

from preprocess import read_wav, stft, mel, _normalize
from scipy.io import wavfile
from scipy import signal

import mindspore as ms

import argparse
import ast
import os

from mindaudio.models import WaveGrad
from hparams import hps


def parse_args():
    parser = argparse.ArgumentParser(description='WaveGrad training')
    parser.add_argument('--device_target', type=str, default="CPU", choices=("GPU", "CPU", 'Ascend'))
    parser.add_argument('--device_id', '-i', type=int, default=0)
    parser.add_argument('--save', '-s', type=str, default='results')
    parser.add_argument('--plot', '-p', type=ast.literal_eval, default=True)
    parser.add_argument('--restore', '-r', type=str, default='')
    parser.add_argument('--restore_url', '-u', type=str, default='')
    parser.add_argument('--wav', '-w', type=str, default='data/LJspeech-1.1/wavs')
    parser.add_argument('--mel', '-m', type=str, default=None)
    parser.add_argument('--data_url', default='')
    parser.add_argument('--train_url', default='')
    args = parser.parse_args()
    return args

args = parse_args()
os.makedirs(args.save, exist_ok=True)
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target=args.device_target, device_id=args.device_id)

if args.mel is not None:
    wav = None
    feature = np.load(args.mel)
    if len(feature.shape) == 2:
        feature = feature[None, ...]
    if feature.shape[-1] == hps.n_mels:
        feature = feature.transpose([0, 2, 1])
    old = args.mel.split('/')[-1].replace('.npy', '')
else:
    wav = read_wav(args.wav)[0]
    feature = stft(wav)
    feature = mel(feature)
    feature = _normalize(feature)[None, ...]
    old = args.wav.split('/')[-1].replace('.wav', '')

gtwav = wav
gtmel = feature[0]

feature = ms.Tensor(feature)
print('old:', old)
print('feature:', feature.shape)

model = WaveGrad(hps)
cur_step = ms.load_checkpoint(args.restore, model, strict_load=True)['cur_step'].asnumpy()
print('restore:', cur_step)

beta = hps.noise_schedule
alpha = 1 - beta
alpha_cum = np.cumprod(alpha).astype(np.float32)
print('alpha_cum:', alpha_cum.shape, alpha_cum.dtype)

audio = np.random.normal(0, 1, [feature.shape[0], hps.hop_samples * feature.shape[-1]]).astype(np.float32)
print('audio:', audio.shape)
noise_scale = ms.Tensor(alpha_cum**0.5)[:, None]

S = len(alpha)
mels = []
wavs = []
for m in tqdm(range(S)):
    n = S - 1 - m
    c1 = 1 / alpha[n]**0.5
    c2 = (1 - alpha[n]) / (1 - alpha_cum[n])**0.5
    pred = model(ms.Tensor(audio), noise_scale[n], feature).asnumpy()
    audio = c1 * (audio - c2 * pred)
    if n > 0:
        noise = np.random.normal(0, 1, audio.shape)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise

    if (m + 1) > (S - 50) and (m + 1) % 5 == 0:
        wavfile.write(args.save + '/%d_predicted_%s_%d.wav' % (cur_step, old, m + 1), hps.sample_rate, audio[0])
        wavs.append(audio[0])
        # de = signal.lfilter([1], [1, -hps.preemph_coef], audio[0]).astype(np.float32)
        # wavfile.write(args.save + '/%d_predicted_de_%s.wav' % (cur_step, old), hps.sample_rate, de)
        # pre = signal.lfilter([1, -hps.preemph_coef], [1], audio[0]).astype(np.float32)
        # wavfile.write(args.save + '/%d_predicted_pre_%s.wav' % (cur_step, old), hps.sample_rate, pre)


# BELOW: plots gif for last steps of reverse diffusion process
if not args.plot:
    exit()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from librosa import display


for wav in wavs:
    feat = _normalize(mel(stft(wav)))
    mels.append(feat)

fig = plt.figure(figsize=(24, 8))
plt.margins(x=0, y=0, tight=True)

titles = ['original %s' % old, 'generated %s' % old]
n_rows = 2
subfigs = fig.subfigures(nrows=n_rows, ncols=1)
axes = []
for i, subfig in enumerate(subfigs):
    ax = subfigs[i].subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    subfigs[i].suptitle(titles[i])
    axes.append(ax)

if gtwav is not None:
    display.waveshow(gtwav, sr=22050, ax=axes[0][0])
axes[0][1].imshow(gtmel, origin="lower")

def plot(wav, mel, frame):
    display.waveshow(wav, sr=22050, ax=axes[1][0], label='%s' % frame)
    axes[1][0].legend()
    img = axes[1][1].imshow(mel, origin="lower")
    return img, 

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

ani.save(
    args.save + '/%d_%s.gif' % (cur_step, old),
    fps=2,
    writer='imagemagick'
)
