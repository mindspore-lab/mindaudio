import numpy as np
from numpy import fft
from scipy.signal import get_window
import mindspore as ms
from mindspore.dataset.audio.utils import BorderType, WindowType, ScaleType, NormType, MelType
import mindspore.dataset.audio as msaudio


__all__ = [
    'amplitude_to_dB',
    'dB_to_amplitude',
    'stft',
    'frame',
    '_pad_center',
    'istft',
    '_window_sumsquare',
    'compute_amplitude',
    'spectrogram',
    'melspectrogram',
    'magphase',
    'melscale',
]


def amplitude_to_dB(S, stype="power", ref=1.0, amin=1e-10, top_db=80.0):
    """
    Turn a spectrogram from the amplitude/power scale to decibel scale.

    Note:
        The dimension of the input spectrogram to be processed should be (..., freq, time).

    Args:
        S (np.ndarray): A power/amplitude spectrogram.
        stype (str, optional): Scale of the input spectrogram, which can be
            'power' or 'magnitude'. Default: 'power'.
        ref (float, callable, optional): Multiplier reference value for generating
            `db_multiplier`. Default: 1.0. The formula is

            :math:`\text{db_multiplier} = Log10(max(\text{ref}, amin))`.

            `amin` refers to the ower bound to clamp the input waveform, which must
            be greater than zero. Default: 1e-10.
        top_db (float, optional): Minimum cut-off decibels, which must be non-negative. Default: 80.0.

    Raises:
        TypeError: If `stype` is not of type 'power' or 'amplitude'.
        TypeError: If `ref` is not of type float.
        ValueError: If `ref` is not a positive number.
        TypeError: If `top_db` is not of type float.
        ValueError: If `top_db` is not a positive number.
        RuntimeError: If input tensor is not in shape of <..., freq, time>.

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.spectrum as spectrum
        >>> waveforms = np.random.random([1, 400 // 2 + 1, 30])
        >>> out = spectrum.amplitude_to_dB(waveforms)
    """
    if np.issubdtype(S.dtype, np.complexfloating):
        raise UserWarning(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(D)**2) instead."
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    multiplier = 10.0 if stype == "power" else 20.0
    db_multiplier = np.log10(max(amin, ref_value))
    specgram_db = multiplier * np.log10(np.clip(magnitude, a_min=amin, a_max=None))
    specgram_db -= multiplier * db_multiplier

    if top_db is not None:
        # expand batch
        shape = specgram_db.shape
        channels = shape[-3] if len(shape) > 2 else 1
        specgram_db = specgram_db.reshape((-1, channels, shape[-2], shape[-1]))

        max_diff = np.amax(specgram_db, axis=(-3, -2, -1)) - top_db
        specgram_db = np.maximum(specgram_db, max_diff.reshape((-1, 1, 1, 1)))

        # Repack batch
        specgram_db = specgram_db.reshape(shape)
    return specgram_db


def dB_to_amplitude(S, ref, power):
    """
        Turn a dB-scaled spectrogram to the power/amplitude scale.

        Args:
            S (np.ndarray): A dB-scaled spectrogram.
            ref (float, callable): Reference which the output will be scaled by. Can be set to be np.max.
            power (float): If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.

        Examples:
            >>> import numpy as np
            >>> import mindaudio.data.spectrum as spectrum
            >>> specgram = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
            >>> out = spectrum.dB_to_amplitude(specgram, 0.5, 0.5)
    """
    if callable(ref):
        ref_value = ref(S)
    else:
        ref_value = np.abs(ref)

    return ref_value * np.power(np.power(10.0, 0.1 * S), power)


def stft(waveforms, n_fft=512, win_length=None, hop_length=None, window="hann", center=True,
         pad_mode="reflect", return_complex=True):
    """
    Short-time Fourier transform (STFT).

    STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short
    overlapping windows.

    Args:
        waveforms (np.ndarray), 1D or 2D array represent the time-serie audio signal.
        n_fft (int): Number of fft point of the STFT. It defines the frequency resolution (n_fft should be <= than
            win_len * (sample_rate/1000)). The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``. In any
            case, we recommend setting ``n_fft`` to a power of two for optimizing the speed of the fast Fourier
            transform (FFT) algorithm.
        win_length (int): Number of frames the sliding window used to compute the STFT. Given the sample rate of the
            audio, the time duration for the windowed signal can be obtained as:
            :math:`duration (ms) = \frac{win_length*1000}{sample_rate}`. Usually, the time duration can be set to
            :math:`~30ms`. If None, win_length will be set to the same as n_fft.
        hop_length (int): Number of frames for the hop of the sliding window used to compute the STFT. If None,
            hop_length will be set to 1/4*n_fft.
        window (str): Name of window function specified for STFT. This function should take an integer (number of
            samples) and outputs an array to be multiplied with each window before fft.
        center (bool): If True (default), the input will be padded on both sides so that the t-th frame is centered at
            time t*hop_length. Otherwise, the t-th frame begins at time t*hop_length.
        pad_mode (str): Padding mode. Options: ["center", "reflect", "constant"]. Default: "reflect".
        return_complex (bool): Whether to return complex array or a real array for the real and imaginary components.

    Returns:
        np.ndarray, STFT

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.spectrum as spectrum
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> matrix = spectrum.stft(waveform)
        (257, 9)
    """
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = win_length // 4

    fft_window = get_window(window, win_length, fftbins=True)
    # Pad the window out to n_fft size
    fft_window = _pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = np.expand_dims(fft_window, axis=-1)

    if center:
        if n_fft > waveforms.shape[-1]:
            raise ValueError(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, waveforms.shape[-1]
                )
            )

        padding = [(0, 0) for _ in range(waveforms.ndim)]
        padding[-1] = (int(n_fft // 2), int(n_fft // 2))
        waveforms = np.pad(waveforms, padding, mode=pad_mode)

    elif n_fft > waveforms.shape[-1]:
        raise ValueError(
            "n_fft={} is too small for input signal of length={}".format(
                n_fft, waveforms.shape[-1]
            )
        )

    data_frames = frame(waveforms, n_fft, hop_length)
    stft_matrix = np.empty((1 + n_fft // 2, data_frames.shape[1]), dtype=np.complex64)

    stft_matrix[..., :] = fft.rfft(fft_window * data_frames[..., :], axis=0)
    if return_complex:
        return stft_matrix
    else:
        return np.stack((stft_matrix.real, stft_matrix.imag), -1)


def frame(x, frame_length=2048, hop_length=64):
    """
    Generate series of frames of the input signal.

    Args:
        x (np.ndarray, Tensor): Input audio signal.
        frame_length (int): The length as to form a group.
        hop_length (int): The hopping length.

    Returns:
        np.ndarray or Tensor, framed signals.
    """
    # x = np.array(x, copy=False)

    if hop_length < 1:
        raise ValueError("Invalid hop_length: {:d}".format(hop_length))

    num_frame = (x.shape[-1] - frame_length) // hop_length + 1
    x_frames = np.zeros(x.shape[:-1] + (frame_length, num_frame))
    if isinstance(x, ms.Tensor):
        x_frames = ms.Tensor(x_frames)
    for i in range(frame_length):
        x_frames[..., i, :] = x[..., i:i + num_frame * hop_length][..., ::hop_length]
    return x_frames


def _pad_center(data, size, axis=-1):
    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths)


def istft(stft_matrix, n_fft=None, win_length=None, hop_length=None, window="hann", center=True):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``data``
    by minimizing the mean squared error between ``stft_matrix`` and STFT of
    ``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT).
    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified ``stft_matrix``.


    Args:
        stft_matrix (np.ndarray): The stft spectrogram.
        n_fft (int or None): Number of fft point of the STFT. By default, it is determined by the number of rows in the
            STFT as the matrix ``D`` is ``(1 + n_fft/2)``.
        win_length (int): Number of frames the sliding window used to compute the STFT. Given the sample rate of the
            audio, the time duration for the windowed signal can be obtained as:
            :math:`duration (ms) = \frac{win_length*1000}{sample_rate}`. Usually, the time duration can be set to
            :math:`~30ms`. If None, win_length will be set to the same as n_fft.
        hop_length (int): Number of frames for the hop of the sliding window used to compute the STFT. If None,
            hop_length will be set to 1/4*n_fft.
        window (str, Callable): Window function specified for STFT. This function should take an integer (number of
            samples) and outputs an array to be multiplied with each window before fft.
        center (bool): If True (default), the input will be padded on both sides so that the t-th frame is centered at
            time t*hop_length. Otherwise, the t-th frame begins at time t*hop_length.

    Returns:
        np.ndarray, the time domain signal.

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.spectrum as spectrum
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> matrix = spectrum.stft(waveform)
        >>> res = spectrum.istft(matrix)
    """

    if n_fft is None:
        n_fft = 2 * (stft_matrix.shape[-2] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    ifft_window = _pad_center(ifft_window, n_fft)
    ifft_window = np.expand_dims(ifft_window, axis=-1)

    n_frames = stft_matrix.shape[-1]

    signal_len = n_fft + hop_length * (n_frames - 1)

    data = np.zeros(shape=stft_matrix.shape[:-2] + (signal_len,), dtype=np.float_)
    data_temp = (ifft_window * fft.irfft(stft_matrix, n=n_fft, axis=-2))

    # we need to deal with the overlap part, instead of directly assign case by case. We can calculate the square sum of
    # windows, assign data_temp to data. Then devide the square sum of windows.
    for i in range(n_frames):
        start = i * hop_length
        data[..., start: min(signal_len, start + n_fft)] += data_temp[..., i]

    # implementation according to librosa
    ifft_window_sum = _window_sumsquare(
        window,
        n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    approx_nonzero_indixes = ifft_window_sum > 1e-9
    data[..., approx_nonzero_indixes] /= ifft_window_sum[approx_nonzero_indixes]

    if center:
        data = data[..., int(n_fft // 2): -int(n_fft // 2)]
    return data


def _window_sumsquare(window, n_frames, win_length, n_fft, hop_length):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=np.float_)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = win_sq ** 2
    win_sq = _pad_center(win_sq, n_fft)

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample: min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def compute_amplitude(waveforms, lengths=None, amp_type="avg", dB=False):
    """Compute amplitude of a batch of waveforms.

    Args:
        waveforms (np.ndarray): A waveform to compute amplitude, shape should be
                                    `[time]` or `[batch, time]` or `[batch, time, channels]`
        lengths (int): The lengths of the waveform excluding the padding. The value should
                                  be `batch`
        amp_type (str["avg", "peak"]): Amplitude type
        dB (bool): Whether to compute amplitude in "dB" scale

    Raises:
        TypeError: If the amplitude type is not supported

    Supported Platforms:
       "GPU"

    Returns
        np.ndarray, the average or peak amplitude of the input waveforms.

    Examples:
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> amp_avg = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='avg')
        >>> amp_peak = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='peak')
        >>> amp_db = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='peak', dB=True)

    """

    if len(waveforms.shape) == 1:
        waveforms = np.expand_dims(waveforms, 0)

    waveforms = np.abs(waveforms)

    if amp_type == "avg":
        if lengths is None:
            out = waveforms.mean(axis=1, keepdims=True)
        else:
            out = waveforms.sum(axis=1, keepdims=True) / lengths
    elif amp_type == "peak":
        out = waveforms.max(axis=1, keepdims=True)
    else:
        raise TypeError("Unsupported amplitude type {}".format(repr(amp_type)))

    if dB:
        out = 20 * np.log10(out)
        return out.clip(min=-80)
    else:
        return out


def spectrogram(waveforms, n_fft=400, win_length=None, hop_length=None, pad=0, window="hann", power=2.0,
                 normalized=False, center=True, pad_mode="reflect", onesided=True):
    """
        Create a spectrogram from an audio signal.

        Args:
            waveforms (np.ndarray): A waveform to compute spectrogram, shape should be
                                    `[time]` or `[batch, time]` or `[batch, time, channels]`
            n_fft (int or None): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
            win_length (int): Window size (default=None, will use n_fft).
            hop_length (int): Length of hop between STFT windows (default=None, will use win_length // 2).
            pad (int): Two sided padding of signal (default=0).
            window (str, Callable): Window function that is applied/multiplied to each frame/window,
                which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann'). Currently kaiser
                window is not supported on macOS.
            power (float): Exponent for the magnitude spectrogram, which must be greater
                than or equal to 0, e.g., 1 for energy, 2 for power, etc. (default=2.0).
            normalized (bool): Whether to normalize by magnitude after stft (default=False).
            center (bool): Whether to pad waveform on both sides (default=True).
            pad_mode (str): Controls the padding method used when center is True,
                which can be 'constant', 'edge', 'reflect', 'symmetric' (default='reflect').
            onesided (bool): Controls whether to return half of results to avoid redundancy (default=True).

        Returns:
            np.ndarray, a spectrogram from an audio signal.

        Exaples：
            >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
            >>> spec = spectrum.spectrogram(waveform)

        """

    win_length = win_length if win_length else n_fft
    hop_length = hop_length if hop_length else win_length // 2
    window = WindowType(window)
    pad_mode = BorderType(pad_mode)
    spectrogram = msaudio.Spectrogram(n_fft, win_length, hop_length, pad, window, power, normalized,
                                    center, pad_mode, onesided)
    return spectrogram(waveforms)


def melspectrogram(waveforms, n_fft=400, win_length=None, hop_length=None, pad=0, window="hann", power=2.0,
                 normalized=False, center=True, pad_mode="reflect", onesided=True, n_mels=128, sample_rate=16000,
                 f_min=0, f_max=None, norm=NormType.NONE, mel_type=MelType.HTK):
    """
    Create a mel-scaled spectrogram from an audio signal.

    Args:
        waveforms (np.ndarray): A waveform to compute melspectrogram, shape should be
                                    `[time]` or `[batch, time]` or `[batch, time, channels]`
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
        win_length (int): Window size (default=None, will use n_fft).
        hop_length (int): Length of hop between STFT windows (default=None, will use win_length // 2).
        pad (int): Two sided padding of signal (default=0).
        window (str, Callable): Window function that is applied/multiplied to each frame/window,
            which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann'). Currently kaiser
            window is not supported on macOS.

        power (float): Exponent for the magnitude spectrogram, which must be greater
            than or equal to 0, e.g., 1 for energy, 2 for power, etc. (default=2.0).
        normalized (bool): Whether to normalize by magnitude after stft (default=False).
        center (bool): Whether to pad waveform on both sides (default=True).
        pad_mode (str): Controls the padding method used when center is True,
            which can be 'constant', 'edge', 'reflect', 'symmetric' (default='reflect').
        onesided (bool): Controls whether to return half of results to avoid redundancy (default=True).
        n_mels (int): Number of mel filterbanks (default=128).
        sample_rate (int): Sample rate of audio signal (default=16000).
        f_min (float): Minimum frequency (default=0).
        f_max (float): Maximum frequency (default=None, will be set to sample_rate // 2).
        norm (str): Type of norm, value should be 'slaney' or 'none'. If norm is 'slaney',
            divide the triangular mel weight by the width of the mel band (default='none').
        mel_type (str): Type of scale to use, value should be 'slaney' or 'htk' (default='htk').

    Returns:
        np.ndarray: Mel frequency spectrogram of size (..., ``n_mels``, time)

    Exaples：
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> spec = spectrum.melspectrogram(waveform)
    """

    win_length = win_length if win_length is not None else n_fft
    hop_length = hop_length if hop_length is not None else win_length // 2

    norm = NormType(norm)
    mel_type = MelType(mel_type)
    window = WindowType(window)
    pad_mode = BorderType(pad_mode)

    spectrogram = msaudio.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, pad=pad, window=window,
                                      power=power, normalized=normalized, center=center, pad_mode=pad_mode, onesided=onesided)

    melscale = msaudio.MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                n_stft=n_fft // 2 + 1, norm=norm, mel_type=mel_type)

    specgram = spectrogram(waveforms)

    return melscale(specgram)


def magphase(s, power, iscomplex):
    """
    Separate a complex-valued spectrogram with shape (..., 2) into its magnitude and phase.

    Args:
        waveform(np.ndarray):A waveform to compute melspectrogram, shape should be
                                    `[time]` or `[batch, time]` or `[batch, time, channels]`
        power (float): Power of the norm, which must be non-negative (default=1.0).
        iscomplex(bool): input is complex or not
    Returns:
        np.ndarray (tuple): A 2-dimension tuple indicating magnitude and phase.

    Examples:
        >>> import numpy as np
        >>> waveforms, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> D = spectrum.stft(waveforms)
        >>> magnitude, phase = spectrum.magphase(D, power=2.0, iscomplex=True)
    """

    if iscomplex:
        mag = np.abs(s)

        #Prevent NaNs and return magnitude 0, phase 1+0j for zero
        zero_to_ones = mag == 0
        mag_nonzero = mag + zero_to_ones
        # Compute real and imaginary seprately, because complex division can produce Nans
        # when denormaliased numbers are involved
        phase = np.empty((s.shape[0], s.shape[1]), dtype=np.complex64)
        phase.real = s.real / mag_nonzero + zero_to_ones
        phase.imag = s.imag / mag_nonzero
        mag **= power
        return mag, phase
    else:
        magphase_from_ms = msaudio.Magphase(power)
        return magphase_from_ms(s)


def melscale(spec, n_mels=128, sample_rate=16000, f_min=0, f_max=None, n_stft=201, norm=NormType.NONE,
        mel_type=MelType.HTK):
    """
        Convert normal STFT to STFT at the Mel scale

        Args:
            n_mels (int) – Number of mel filterbanks (default=128).
            sample_rate (int) – Sample rate of audio signal (default=16000).
            f_min (float) – Minimum frequency (default=0).
            f_max (float) – Maximum frequency (default=None, will be set to sample_rate // 2).
            n_stft (int) – Number of bins in STFT (default=201).
            norm (NormType) – Type of norm, value should be NormType.SLANEY or NormType::NONE. If norm is NormType.
            SLANEY, divide the triangular mel weight by the width of the mel band. (default=NormType.NONE).
            mel_type (MelType) – Type to use, value should be MelType.SLANEY or MelType.HTK (default=MelType.HTK).
        Returns:
            np.ndarray (tuple): A 2-dimension tuple indicating magnitude and phase.

        Examples:
            >>> import numpy as np
            >>> import mindaudio.data.spectrum as spectrum
            >>> waveforms, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
            >>> spec = spectrum.spectrogram(waveforms, n_fft=1024)
            >>> melscale_spec = spectrum.melscale(spec, n_stft=1024 // 2 +1)
        """
    f_max = f_max if f_max is not None else sample_rate // 2
    mel_scale = msaudio.MelScale(n_mels, sample_rate, f_min, f_max, n_stft, norm, mel_type)
    return mel_scale(spec)

