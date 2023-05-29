import mindspore as ms
import mindspore.dataset.audio as msaudio
import numpy as np
from mindspore.dataset.audio.utils import BorderType, MelType, NormType, WindowType
from numpy import fft
from scipy.signal import get_window

__all__ = [
    "amplitude_to_dB",
    "dB_to_amplitude",
    "stft",
    "istft",
    "compute_amplitude",
    "spectrogram",
    "melspectrogram",
    "magphase",
    "melscale",
    "resynthesize",
]

# Define max block sizes(256 KB)
MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


def amplitude_to_dB(wavform, stype="power", ref=1.0, amin=1e-10, top_db=80.0):
    """
    Turn a spectrogram from the amplitude/power scale to decibel scale.

    Note:
        The dimension of the input spectrogram to be processed should be (..., freq, time).

    Args:
        wavform (np.ndarray): A power/amplitude spectrogram.
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
    if np.issubdtype(wavform.dtype, np.complexfloating):
        raise UserWarning(
            "amplitude_to_db was called on complex input so phase "
            "information will be discarded. To suppress this warning, "
            "call amplitude_to_db(np.abs(D)**2) instead."
        )
        magnitude = np.abs(wavform)
    else:
        magnitude = wavform

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


def dB_to_amplitude(wavform, ref, power):
    """
    Turn a dB-scaled spectrogram to the power/amplitude scale.

    Args:
        wavform (np.ndarray): A dB-scaled spectrogram.
        ref (float, callable): Reference which the output will be scaled by. Can be set to be np.max.
        power (float): If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.spectrum as spectrum
        >>> specgram = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> out = spectrum.dB_to_amplitude(specgram, 0.5, 0.5)
    """
    if callable(ref):
        ref_value = ref(wavform)
    else:
        ref_value = np.abs(ref)

    return ref_value * np.power(np.power(10.0, 0.1 * wavform), power)


def _expand_to(x, ndim, axes):
    axes_tup = tuple([axes])  # type: ignore

    shape = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]
    return x.reshape(shape)


def stft(
    waveforms,
    n_fft=512,
    win_length=None,
    hop_length=None,
    window="hann",
    center=True,
    pad_mode="constant",
    return_complex=True,
):
    """
    Short-time Fourier transform (STFT).

    STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short
    overlapping windows.

    Args:
        waveforms (np.ndarray), 1D or 2D array represent the time-serie audio signal.
        n_fft (int): Number of fft point of the STFT. It defines the frequency resolution. The number of rows in
            the STFT matrix ``D`` is ``(1 + n_fft/2)``.
            Notes:n_fft = 2 ** n, n_fft <= win_len * (sample_rate/1000)
        win_length (int): Number of frames the sliding window used to compute the STFT.
            Notes:duration (ms) = {win_length*1000}{sample_rate} If None, win_length = n_fft.
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
    fft_window = _expand_to(fft_window, ndim=1 + waveforms.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > waveforms.shape[-1]:
            raise ValueError(
                "n_fft={} is too small for input signal of length={}".format(
                    n_fft, waveforms.shape[-1]
                )
            )

        # Set up the padding array to be empty, and we'll fix the target dimension later
        padding = [(0, 0) for _ in range(waveforms.ndim)]

        # How many frames depend on left padding?
        start_k = int(np.ceil(n_fft // 2 / hop_length))

        # What's the first frame that depends on extra right-padding?
        tail_k = (waveforms.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

        if tail_k <= start_k:
            # If tail and head overlap, then just copy-pad the signal and carry on
            start = 0
            extra = 0
            padding[-1] = (n_fft // 2, n_fft // 2)
            waveforms = np.pad(waveforms, padding, mode=pad_mode)
        else:
            # If tail and head do not overlap, then we can implement padding on each part separately
            # and avoid a full copy-pad

            # "Middle" of the signal starts here, and does not depend on head padding
            start = start_k * hop_length - n_fft // 2
            padding[-1] = (n_fft // 2, 0)

            waveforms_pre = np.pad(
                waveforms[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
                padding,
                mode=pad_mode,
            )
            af_frames = frame(waveforms_pre, frame_length=n_fft, hop_length=hop_length)[
                ..., :start_k
            ]
            the_shape_of_frames = af_frames.shape
            extra = the_shape_of_frames[-1]

            if (
                tail_k * hop_length - n_fft // 2 + n_fft
                <= waveforms.shape[-1] + n_fft // 2
            ):
                padding[-1] = (0, n_fft // 2)
                y_post = np.pad(
                    waveforms[..., (tail_k) * hop_length - n_fft // 2 :],
                    padding,
                    mode=pad_mode,
                )
                y_frames_post = frame(y_post, frame_length=n_fft, hop_length=hop_length)
                extra += y_frames_post.shape[-1]
            else:
                # the end padding
                post_shape = list(the_shape_of_frames.shape)
                post_shape[-1] = 0
                y_frames_post = np.empty_like(af_frames, shape=post_shape)
    else:
        start = 0
        extra = 0
        if n_fft > waveforms.shape[-1]:
            raise ValueError(
                f"n_fft={n_fft} is too large for uncentered analysis of input signal of length={waveforms.shape[-1]}"
            )
    # Window the time series.
    y_frames = frame(waveforms[..., start:], frame_length=n_fft, hop_length=hop_length)
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    shape[-1] += extra
    stft_matrix = np.empty(shape, order="F", dtype=np.complex64)

    # Fill in the warm-up
    if center and extra > 0:
        off_start = af_frames.shape[-1]
        stft_matrix[..., :off_start] = fft.rfft(fft_window * af_frames, axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0

    n_columns = max(
        int(MAX_MEM_BLOCK // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize)), 1
    )

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])
        stft_matrix[..., bl_s + off_start : bl_t + off_start] = fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )

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
        x_frames[..., i, :] = x[..., i : i + num_frame * hop_length][..., ::hop_length]
    return x_frames


def _pad_shape(y_shift, data_shape):
    need_shape = y_shift.shape[-1]

    if need_shape > data_shape:
        slices = [slice(None)] * y_shift.ndim
        slices[-1] = slice(0, data_shape)
        return y_shift[tuple(slices)]

    elif need_shape < data_shape:
        lengths = [(0, 0)] * y_shift.ndim
        lengths[-1] = (0, data_shape - need_shape)
        return np.pad(y_shift, lengths, mode="constant")

    return y_shift


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


def overlap_add(output_buffer, frames, hop_length):
    n_fft = frames.shape[-2]
    for sample_frame in range(frames.shape[-1]):
        sample = sample_frame * hop_length
        output_buffer[..., sample : (sample + n_fft)] += frames[..., sample_frame]


def istft(
    stft_matrix,
    n_fft=None,
    win_length=None,
    hop_length=None,
    window="hann",
    center=True,
    length=None,
):
    # pylint: disable=C,R,W,E,F
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
        length (int): int > 0, optional, If provided, the output ``y`` is zero-padded or clipped to exactly
            ``length`` samples.

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

    # Pad out to match n_fft, and add broadcasting axes
    ifft_window = _pad_center(ifft_window, size=n_fft)
    ifft_window = np.expand_dims(ifft_window, axis=-1)

    # For efficiency, trim STFT frames according to signal length if available
    if length:
        if center:
            padded_length = length + int(n_fft)
        else:
            padded_length = length
        n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))
    else:
        n_frames = stft_matrix.shape[-1]

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    shape.append(expected_signal_len)
    y = np.zeros(shape, dtype=np.float_)

    n_columns = (
        2 ** 8 * 2 ** 10 // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize)
    )
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(0, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        overlap_add(y[..., frame * hop_length :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = _window_sumsquare(
        window=window,
        n_frames=n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    approx_nonzero_indices = ifft_window_sum > 1e-9
    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[..., int(n_fft // 2) : -int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = _pad_shape(y[..., start:], data_shape=length)

    return y


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
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
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


def spectrogram(
    waveforms,
    n_fft=400,
    win_length=None,
    hop_length=None,
    pad=0,
    window="hann",
    power=2.0,
    normalized=False,
    center=True,
    pad_mode="reflect",
    onesided=True,
):
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
            which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann').
            Currently kaiser window is not supported on macOS.
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
    spectrogram = msaudio.Spectrogram(
        n_fft,
        win_length,
        hop_length,
        pad,
        window,
        power,
        normalized,
        center,
        pad_mode,
        onesided,
    )
    return spectrogram(waveforms)


def melspectrogram(
    waveforms,
    n_fft=400,
    win_length=None,
    hop_length=None,
    pad=0,
    window="hann",
    power=2.0,
    normalized=False,
    center=True,
    pad_mode="reflect",
    onesided=True,
    n_mels=128,
    sample_rate=16000,
    f_min=0,
    f_max=None,
    norm=NormType.NONE,
    mel_type=MelType.HTK,
):
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
            which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann').
            Currently kaiser window is not supported on macOS.

        power (float): Exponent for the magnitude spectrogram, which must be
            greater than or equal to 0, e.g., 1 for energy, 2 for power, etc. (default=2.0).
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

    spectrogram = msaudio.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        pad=pad,
        window=window,
        power=power,
        normalized=normalized,
        center=center,
        pad_mode=pad_mode,
        onesided=onesided,
    )

    melscale = msaudio.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_stft=n_fft // 2 + 1,
        norm=norm,
        mel_type=mel_type,
    )

    specgram = spectrogram(waveforms)

    return melscale(specgram)


def magphase(waveform, power, iscomplex=True):
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
        mag = np.abs(waveform)

        # Prevent NaNs and return magnitude 0, phase 1+0j for zero
        zero_to_ones = mag == 0
        mag_nonzero = mag + zero_to_ones
        # Compute real and imaginary seprately, because complex division can produce Nans
        # when denormaliased numbers are involved
        phase = np.empty((waveform.shape[0], waveform.shape[1]), dtype=np.complex64)
        phase.real = waveform.real / mag_nonzero + zero_to_ones
        phase.imag = waveform.imag / mag_nonzero
        mag **= power
        return mag, phase
    else:
        magphase_from_ms = msaudio.Magphase(power)
        return magphase_from_ms(waveform)


def melscale(
    spec,
    n_mels=128,
    sample_rate=16000,
    f_min=0,
    f_max=None,
    n_stft=201,
    norm=NormType.NONE,
    mel_type=MelType.HTK,
):
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
    mel_scale = msaudio.MelScale(
        n_mels, sample_rate, f_min, f_max, n_stft, norm, mel_type
    )
    return mel_scale(spec)


def resynthesize(enhanced_mag, noisy_inputs, normalize_wavs=True):
    """Function for resynthesizing waveforms from enhanced mags.

    Arguments:
        enhanced_mag (np.ndarray): Predicted spectral magnitude, should be two dimensional.
        noisy_inputs (np.ndarray): The noisy waveforms before any processing, to extract phase.
        normalize_wavs (bool): Whether to normalize the output wavs before returning them.

    Returns:
        enhanced_wav (np.ndarray): The resynthesized waveforms of the enhanced magnitudes with noisy phase.

    Examples:
        >>> waveform1, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> D = spectrum.stft(waveform1, return_complex=False)
        >>> mag, _ = spectrum.magphase(D, power=1.0, iscomplex=False)
        >>> enhanced_wav = spectrum.resynthesize(mag, waveform1, normalize_wavs=False)
    """

    # To extract phase of input noisy
    noisy_feats = stft(noisy_inputs, return_complex=False)
    noisy_phase = np.arctan2(noisy_feats[:, :, 1], noisy_feats[:, :, 0])

    pre_stack = np.stack([np.cos(noisy_phase), np.sin(noisy_phase),], axis=-1,)
    # using enhanced magnitude to combine data
    complex_predictions = np.expand_dims(enhanced_mag, -1) * pre_stack
    result = complex_predictions[:, :, 0] + 1j * complex_predictions[:, :, 1]

    pred_wavs = istft(result)

    # peaked amplitudes, ignore lengths, need to normalize.
    if normalize_wavs:
        from .processing import normalize

        pred_wavs = normalize(pred_wavs, norm="max")

    return pred_wavs
