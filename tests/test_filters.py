import numpy as np
import mindaudio.data.io as io
import mindaudio.data.filters as filters
from mindaudio.data.augment import convolve1d


def test_notch_filter():
    waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
    kernel = filters.notch_filter(0.25)
    notched_signals = convolve1d(waveform, kernel)
    print(notched_signals.shape)


def test_low_pass_filter():
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])

    sample_rate = 44100
    cutoff_freq = 1500
    out_waveform = filters.low_pass_filter(waveform, sample_rate, cutoff_freq)
    print(out_waveform)


def test_peaking_equalizer():
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])

    sample_rate = 44100
    center_freq = 1500
    gain = 3.0
    quality_factor = 0.707
    out_waveform = filters.peaking_equalizer(waveform, sample_rate, center_freq, gain, quality_factor)
    print(out_waveform)


if __name__ == "__main__":
    test_notch_filter()
    test_low_pass_filter()
    test_peaking_equalizer()