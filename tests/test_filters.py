import mindaudio.data.io as io
import mindaudio.data.filters as filters
from mindaudio.data.augment import convolve1d


def test_notch_filter():
    waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
    kernel = filters.notch_filter(0.25)
    notched_signals = convolve1d(waveform, kernel)
    print(notched_signals.shape)


if __name__ == "__main__":
    test_notch_filter()