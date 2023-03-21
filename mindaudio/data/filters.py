import numpy as np
from typing import Optional, Union
from typing_extensions import Literal
import mindspore.dataset.audio as msaudio


__all__ = [
    'notch_filter',
    'low_pass_filter',
    'peaking_equalizer',
    'contrast',
    'riaa_biquad',
    'treble_biquad',
    'dcshift',
    'filtfilt',
    'mel'
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
    assert filter_width % 2 != 0
    assert 0 < notch_freq <= 1

    pad = filter_width // 2
    notch_freq += notch_width
    inputs = np.arange(filter_width) - pad

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
    # pylint: disable=C,R,W,E,F
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
    # pylint: disable=C,R,W,E,F
    """
    Allows audio signals with a frequency lower than the given cutoff to pass through
    and attenuates signals with frequencies higher than the cutoff frequency.

    Args:
        waveform(np.ndarray): A batch of data in shape (n,) or (n, n_channel).
        sample_rate: the audio sample rate of the inputted audio.
        cutoff_freq: frequency (in Hz) where signals with higher frequencies will
            begin to be reduced by 6dB per octave (doubling in frequency) above this point.

    Returns:
        np.ndarray, the waveform after low pass equalizer.

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
    # pylint: disable=C,R,W,E,F
    """
    Applies a two-pole peaking equalization filter. The signal-level at and around
    `center_freq` can be increased or decreased, while all other frequencies are unchanged

    Args:
        waveform(np.ndarray): A batch of data in shape (n,) or (n, n_channel).
        sample_rate: the audio sample rate of the inputted audio.
        center_freq: point in the frequency spectrum at which EQ is applied.
        gain: amount of gain (boost) or reduction (cut) that is applied at a
            given frequency. Beware of clipping when using positive gain.
        q: ratio of center frequency to bandwidth; bandwidth is inversely
            proportional to Q, meaning that as you raise Q, you narrow the bandwidth.

    Returns:
        np.ndarray, the waveform after peaking equalizer

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


def contrast(waveform, enhancement_amount=75.0):
    """
    Apply contrast effect for audio waveform.
    Comparable with compression, this effect modifies an audio signal to make it sound louder.
    Similar to `SoX <http://sox.sourceforge.net/sox.html>`_ implementation.

    Args:
        waveform(np.ndarray): The dimension of the audio waveform to be processed needs to be (..., time).
        enhancement_amount (float, optional): Controls the amount of the enhancement,in range of [0, 100]. Default:
            75.0. Note that `enhancement_amount` equal to 0 still gives a significant contrast enhancement

    Returns:
        contrast_wav(np.ndarray):The dimension of the audio waveform is (..., time)

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.filters as filters
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> contrast_wav = filters.contrast(waveform)

    """
    effect = msaudio.Contrast(enhancement_amount)
    contrast_wav = effect(waveform)
    return contrast_wav


def riaa_biquad(waveform, sample_rate=44100):
    """
    Apply RIAA vinyl playback equalization.
    Similar to `SoX <http://sox.sourceforge.net/sox.html>`_ implementation.

    Args:
        waveform(np.ndarray): The dimension of the audio waveform to be processed needs to be (..., time).
        sample_rate (int): sampling rate of the waveform, e.g. 44100 (Hz), can only be one of 44100, 48000, 88200,
            96000.

    Returns:
        riaa_wav(np.ndarray): The dimension of the audio waveform is (..., time)

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.filters as filters
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> riaa_wav = filters.riaa_biquad(waveform)
    """
    effect = msaudio.RiaaBiquad(sample_rate)
    riaa_wav = effect(waveform)
    return riaa_wav


def treble_biquad(waveform, sample_rate, gain, central_freq=3000, Q=0.707):
    """
    Design a treble tone-control effect.
    Similar to `SoX <http://sox.sourceforge.net/sox.html>`_ implementation.

    Args:
        waveform(np.ndarray): The dimension of the audio waveform to be processed needs to be (..., time).
        sample_rate (int): Sampling rate (in Hz), which can't be zero.
        gain (float): Desired gain at the boost (or attenuation) in dB.
        central_freq (float): Central frequency (in Hz). Default: 3000.
        Q (float): `Quality factor <https://en.wikipedia.org/wiki/Q_factor>`_ ,in range of (0, 1]. Default: 0.707.

    Returns:
        treble_wav(np.ndarray): The dimension of the audio waveform is (..., time)

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.filters as filters
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> treble_wav = filters.treble_biquad(waveform, sample_rate=44100, gain=5)
    """
    effect = msaudio.TrebleBiquad(sample_rate, gain, central_freq, Q)
    treble_wav = effect(waveform)
    return treble_wav


def dcshift(waveform, shift, limiter_gain):
    """
    Apply a DC shift to the audio. This can be useful to remove DC offset from audio.

    Args:
        waveform(np.ndarray): The dimension of the audio waveform to be processed needs to be (..., time).
        shift (float): The amount to shift the audio, the value must be in the range [-2.0, 2.0].
        limiter_gain (float, optional): Used only on peaks to prevent clipping, the value should be much less than 1,
            such as 0.05 or 0.02.

    Returns:
        shifted_wav(np.ndarray): The dimension of the audio waveform is (..., time)

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.filters as filters
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> treble_wav = filters.dcshift(waveform, shift=0.5, limiter_gain=0.02)
    """
    effect = msaudio.DCShift(shift, limiter_gain)
    shifted_wav = effect(waveform)
    return shifted_wav


def filtfilt(waveform, N, Wn, btype):
    """
    Apply a DC shift to the audio. This can be useful to remove DC offset from audio.

    Args:
        waveform(np.ndarray): The dimension of the audio waveform to be processed needs to be (..., time).
        N(int): The order of the filter.
        Wn(float): Normalized cutoff frequency. The formula Wn=2* cutoff_frequency / sample_rate.
        btype(str): {‘lowpass', ‘highpass', ‘bandpass', ‘bandstop'}.

    Returns:
        filted_wav(np.ndarray): The dimension of the audio waveform is (..., time)

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.filters as filters
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> treble_wav = filters.filtfilt(waveform, N=8, Wn=0.02, btype='highpass')
    """
    from scipy import signal

    b, a = signal.butter(N, Wn, btype)
    filted_wav = signal.filtfilt(b, a, waveform)
    return filted_wav


def hz_to_mel(frequencies, htk=False):
    frequencies = np.asanyarray(frequencies)
    if htk:
        mels: np.ndarray = 2595.0 * np.log10(1.0 + frequencies / 700.0)
        return mels
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    #log-scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    #linear scaling
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    hz = mel_to_hz(mels, htk=htk)
    return hz


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None,
        norm: Optional[Union[Literal["slaney"], float]] = "slaney"):
    """Create a Mel filter-bank.
    This produces a linear transformation matrix to project FFT bins onto Mel-frequency bins.

    Args:
        sr(int): sampling rate of the incoming signal
        n_fft(int): number of FFT components
        n_mels(int): number of Mel bands to generate
        fmin(float): lowest frequency (in Hz)
        fmax(float): highest frequency (in Hz).If `None`, use ``fmax = sr / 2.0``
        norm({None, 'slaney', or number} [scalar]): If 'slaney', divide the triangular mel weights by the width of the
        mel band(area normalization).If numeric, use `librosa.util.normalize` to normalize each filter by to unit
        l_p norm.

    Returns:
        M (np.ndarray): [shape=(n_mels, 1 + n_fft/2)] Mel transform matrix

    Examples:
        >>> import mindaudio.data.filters as filters
        >>> melfb = filters.mel(sr=22050, n_fft=2048)
    """

    if fmax is None:
        fmax = float(sr) / 2

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_freqs = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax)

    fdiff = np.diff(mel_freqs)
    ramps = np.subtract.outer(mel_freqs, fftfreqs)

    for index in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[index] / fdiff[index]
        upper = ramps[index + 2] / fdiff[index + 1]

        # .. then intersect them with each other and zero
        weights[index] = np.maximum(0, np.minimum(lower, upper))

    if isinstance(norm, str):
        if norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_freqs[2: n_mels + 2] - mel_freqs[:n_mels])
            weights *= enorm[:, np.newaxis]
    else:
        import mindaudio.data.processing as processing
        weights = processing.normalize(weights, norm=norm, axis=-1)

    return weights
