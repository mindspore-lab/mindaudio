import numpy as np


__all__ = [
    'notch_filter',
]


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """
    A notch filter only filters a very narrow band.

    Args:
        notch_freq (float): Frequency to put notch as a fraction of sample rate / 2. The range of possibile
        filter_width (int): Filter width in samples determing the width of pass band. Longer filters have smaller
            transition bands.
        notch_width (float): Width of the notch, as a fraction of the sampling rate / 2.

    Returns:
        np.ndarray, notch filter kernel.

     Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.filters as filters
        >>> from mindaudio.data.augment import convolve1d
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> kernel = filters.notch_filter(0.25)
        >>> notched_signals = convolve1d(waveform, kernel)
    """
    assert 0 < notch_freq <= 1
    assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = np.arange(filter_width) - pad
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        # The zero is at the middle index
        return np.concatenate([np.sin(x[:pad]) / x[:pad], np.ones(1), np.sin(x[pad + 1:]) / x[pad + 1:]])

    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= np.blackman(filter_width + 1)[:-1]
    hlpf /= np.sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= np.blackman(filter_width + 1)[:-1]
    hhpf /= -np.sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).reshape(1, -1, 1)
