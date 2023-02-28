# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Compute FBank features."""

import math
import numpy as np
from mindaudio.data.io import read


def get_waveform(batch, sample_rate=16000):
    """Load audios.

    Args:
        batch (list): a list of (uttid, wav_path, labels)
        sample_rate (int): sample rate of audios. Defaults to 16000.

    Return:
        tuple: (uttids, wav_samples, wav_lengths, labels)
    """
    uttids = []
    wavs = []
    lengths = []
    for _, x in enumerate(batch):
        wav_path = x[1]
        waveform, sample_rate = read(wav_path)
        waveform = waveform * (1 << 15)
        uttids.append(x[0])
        wavs.append(waveform)
        lengths.append(waveform.shape[0])

    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_uttids = [uttids[i] for i in order]
    sorted_wavs = [wavs[i] for i in order]
    sorted_lengths = [lengths[i] for i in order]
    labels = [x[2].split() for x in batch]
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in order]

    return sorted_uttids, sorted_wavs, sorted_lengths, sorted_labels


def inverse_mel_scale(mel_freq):
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq/700.0)


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq/700.0)


def get_mel_banks(num_bins: int, window_length_padded: int, sample_freq: float, low_freq: float, high_freq: float):
    """Get mel banks for extracting features.

    Args:
        num_bins (int): number of mel bins.
        window_length_padded (int): length of windows to split frame.
        sample_freq (int): sample rate of audios.
        low_freq (float): lowest frequency.
        high_freq (float): highest frequency.
    """

    num_fft_bins = window_length_padded // 2

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq-mel_low_freq) / (num_bins+1)

    bins = np.arange(num_bins).reshape(-1, 1)
    left_mel = mel_low_freq + bins*mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bins+1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bins+2.0) * mel_freq_delta  # size(num_bins, 1)

    center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    # mel = mel_scale(fft_bin_width * np.arange(num_fft_bins)).unsqueeze(0)
    mel = np.expand_dims(mel_scale(fft_bin_width * np.arange(num_fft_bins)), 0)
    # size (num_bins, num_fft_bins)
    up_slope = (mel-left_mel) / (center_mel-left_mel)
    down_slope = (right_mel-mel) / (right_mel-center_mel)
    # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
    feat = np.where(up_slope > down_slope, down_slope, up_slope)
    feat = np.where(feat < 0, 0, feat)
    feat = np.pad(feat, ((0, 0), (0, 1)), 'constant')

    return feat, center_freqs


# Enframe with Hamming window function
def preemphasis(signal):
    """Perform preemphasis on the input signal."""
    return np.append(signal[0], signal[1:] - 0.97 * signal[:-1])


def enframe(signal, frame_len, frame_shift):
    """Enframe with Hamming widow function."""

    num_samples = signal.size
    win = np.power(np.hanning(frame_len), 0.85)
    num_frames = np.floor((num_samples-frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift:i*frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win
    return frames


def get_spectrum(frames, fft_len):
    """Get spectrum using fft."""
    c_fft = np.fft.rfft(frames, n=fft_len)
    spectrum = np.abs(c_fft)**2
    return spectrum


def fbank(spectrum, num_filter, fs):
    """Get mel filter bank features from spectrum.

    args:
        spectrum: a num_frames by fft_len/2+1 array(real)
        num_filter: mel filters number, default 23

    return:
        fbank feature, a num_frames by num_filter array
    """
    mel_energies = get_mel_banks(num_filter, 512, fs * 2, 20, 8000)[0]
    feats = np.dot(spectrum, mel_energies.T)
    feats = np.where(feats == 0, np.finfo(float).eps, feats)
    feats = np.log(feats)

    return feats


def compute_fbank_feats(wav, sample_rate, frame_len, frame_shift, mel_bin):
    """compute fbank feats."""
    signal = preemphasis(wav)
    frame_len = sample_rate * frame_len // 1000
    frame_shift = sample_rate * frame_shift // 1000
    frames = enframe(signal, frame_len=frame_len, frame_shift=frame_shift)
    frames -= np.mean(frames)
    spectrum = get_spectrum(frames, fft_len=512)
    fbank_feats = fbank(spectrum, num_filter=mel_bin, fs=sample_rate / 2)
    return fbank_feats
