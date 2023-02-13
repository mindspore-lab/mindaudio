import numpy as np


__all__ = [
    'notch_filter',
    'low_pass_filter',
    'peaking_equalizer',
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


def cal_filter_by_coffs(waveform, b, a):
    if waveform.ndim == 1:
        changed_waveform = waveform
        o2 = 0.
        o1 = 0.
        i2 = 0.
        i1 = 0.
        j = 0
        while j < waveform.shape[-1]:
            o0 = waveform[j] * b[0] + i1 * b[1] + i2 * b[2] - o1 * a[1] - o2 * a[2]
            i2 = i1
            i1 = waveform[j]
            o2 = o1
            o1 = o0
            changed_waveform[j] = min(o0, 1.0)
            j += 1
    else:
        waveform = waveform.T
        changed_waveform = waveform
        i = 0
        while i < waveform.shape[0]:
            j = 0
            o2 = 0.
            o1 = 0.
            i2 = 0.
            i1 = 0.
            while j < waveform.shape[1]:
                o0 = waveform[i][j] * b[0] + i1 * b[1] + i2 * b[2] - o1 * a[1] - o2 * a[2]
                i2 = i1
                i1 = waveform[i][j]
                o2 = o1
                o1 = o0
                changed_waveform[i][j] = min(o0, 1.0)
                j += 1
            i += 1
        changed_waveform = changed_waveform.T
    return changed_waveform


def low_pass_filter(waveform, sample_rate, cutoff_freq):
    """
    Allows audio signals with a frequency lower than the given cutoff to pass through
    and attenuates signals with frequencies higher than the cutoff frequency
    @param waveform: the path to the audio or a variable of type np.ndarray that
        will be augmented
    @param sample_rate: the audio sample rate of the inputted audio
    @param cutoff_freq: frequency (in Hz) where signals with higher frequencies will
        begin to be reduced by 6dB per octave (doubling in frequency) above this point

    @returns: np.ndarray, the waveform after low pass equalizer

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.filters as filters
        >>> waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
        >>>                     [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
        >>>
        >>> sample_rate = 44100
        >>> center_freq = 1500
        >>> out_waveform = filters.low_pass_filter(waveform, sample_rate, center_freq)
    """
    q = 0.707
    w0 = 2 * np.pi * cutoff_freq / sample_rate
    alpha = np.sin(w0) / (2 * q)

    b0 = (1 - np.cos(w0)) / 2
    b1 = 1 - np.cos(w0)
    b2 = (1 - np.cos(w0)) / 2
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([a0, a1 / a0, a2 / a0])

    return cal_filter_by_coffs(waveform, b, a)


def peaking_equalizer(waveform, sample_rate, center_freq, gain, q=0.707):
    """
    Applies a two-pole peaking equalization filter. The signal-level at and around
    `center_hz` can be increased or decreased, while all other frequencies are unchanged
    @param waveform: the path to the audio or a variable of type np.ndarray that
        will be augmented
    @param sample_rate: the audio sample rate of the inputted audio
    @param center_freq: point in the frequency spectrum at which EQ is applied
    @param q: ratio of center frequency to bandwidth; bandwidth is inversely
        proportional to Q, meaning that as you raise Q, you narrow the bandwidth
    @param gain: amount of gain (boost) or reduction (cut) that is applied at a
        given frequency. Beware of clipping when using positive gain

    @returns: np.ndarray, the waveform after peaking equalizer

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.filters as filters
        >>> waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
        >>>                     [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
        >>>
        >>> sample_rate = 44100
        >>> center_freq = 1500
        >>> gain = 3.0
        >>> quality_factor = 0.707
        >>> out_waveform = filters.peaking_equalizer(waveform, sample_rate, center_freq, gain, quality_factor)
    """
    aa = np.exp(gain / 40 * np.log(10.))
    w0 = 2 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * aa
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * aa
    a0 = 1 + alpha / aa
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / aa

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([a0, a1 / a0, a2 / a0])

    return cal_filter_by_coffs(waveform, b, a)
