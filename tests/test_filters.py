import numpy as np
import sys
sys.path.append('.')
import mindaudio.data.io as io
import mindaudio.data.filters as filters
from mindaudio.data.augment import convolve1d


def test_notch_filter():
    waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
    kernel = filters.notch_filter(0.25)
    notched_signals = convolve1d(waveform, kernel)
    print(notched_signals.shape)


def test_low_pass_filter():
    waveform, sample_rate = io.read('./samples/ASR/BAC009S0002W0122.wav')
    cutoff_freq = 1500
    out_waveform = filters.low_pass_filter(waveform, sample_rate, cutoff_freq)
    print(out_waveform)


def test_peaking_equalizer():
    waveform, sample_rate = io.read('./samples/ASR/BAC009S0002W0122.wav')
    center_freq = 1500
    gain = 3.0
    quality_factor = 0.707
    out_waveform = filters.peaking_equalizer(waveform, sample_rate, center_freq, gain, quality_factor)
    print(out_waveform)


if __name__ == "__main__":
    test_notch_filter()
    test_low_pass_filter()
    test_peaking_equalizer()