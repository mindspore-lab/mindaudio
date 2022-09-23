import numpy as np
from numpy import fft
from scipy.signal import get_window
import mindspore as ms
import mindspore._c_dataengine as cde
from mindspore.dataset.audio.utils import ScaleType
from mindspore.nn import Conv1d


__all__ = [
    'amplitude_to_dB',
    'dB_to_amplitude',
    'stft',
    'frame',
    '_pad_center',
    'istft',
    '_window_sumsquare',
    'compute_amplitude',
    'normalize',
    'unitarize',
    'rescale',
    'stereo_to_mono',
    'notch_filter',
    'trim',
    'split',
    'reverberate',
    'convolve1d',
]


# Todo: converts all the operations in MindSpore to achieve a better performance.
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
        >>> waveforms = np.random.random([1, 400 // 2 + 1, 30])
        >>> out = amplitude_to_dB(waveforms)
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
            >>> specgram = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
            >>> out = dB_to_amplitude(specgram, 0.5, 0.5)
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
        >>> data = np.arange(1024)
        >>> stft(waveforms).shape
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
        >>> array = np.sin(np.arange(16000.0))
        >>> waveforms = np.expand_dims(array, 0)
        >>> amp_avg = compute_amplitude(waveforms, lengths=waveforms.shape[1], amp_type='avg')
        >>> [[0.6366068]]
        >>> amp_peak = compute_amplitude(waveforms, lengths=waveforms.shape[1], amp_type='peak')
        >>> [[0.9999953]]
        >>> amp_db = compute_amplitude(waveforms, lengths=waveforms.shape[1], amp_type='peak', db=True)
        >>> [[-4.0899922e-05]]

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


def normalize(waveforms, norm="max", axis=0):
    """Normalize an array along a specified axis.

    Args:
        waveforms (np.ndarray): An audio signal to normalize
        norm (Union['min', 'max', 'mean', 'mean_std', 'l0', 'l1', 'l2']): Normalization type, here "max" means l-infinity norm,
                                                                          "mean_std" refers to the mean standard deviation norm
        axis (int): Axis along which to compute the norm

    Raises:
        TypeError: If the normalization type is not supported

    Supported Platforms:
       "GPU"

    Returns:
        np.ndarray, normalized array.

    Examples:
        >>> # Construct an example matrix
        >>> waveforms = np.vander(np.arrange(-2, 2))
        >>> # Max (L-Infinity)-normalize the rows
        >>> normalize(waveforms, axis=1)
        array([[-1.   ,  0.5  , -0.25 ,  0.125],
               [-1.   ,  1.   , -1.   ,  1.   ],
               [ 0.   ,  0.   ,  0.   ,  1.   ],
               [ 1.   ,  1.   ,  1.   ,  1.   ]])
        >>> # Min-normalize the columns
        >>> normalize(waveforms, norm="min")
        array([[-8.   ,  4.   , -2.   ,  1.   ],
               [-1.   ,  1.   , -1.   ,  1.   ],
               [ 0.   ,  0.   ,  0.   ,  1.   ],
               [ 1.   ,  1.   ,  1.   ,  1.   ]])
        >>> # l0-normalize the columns
        >>> normalize(waveforms, norm="l0")
        array([[-2.667,  1.333, -0.667,  0.25 ],
               [-0.333,  0.333, -0.333,  0.25 ],
               [ 0.   ,  0.   ,  0.   ,  0.25 ],
               [ 0.333,  0.333,  0.333,  0.25 ]])
        >>> # l1-normalize the columns
        >>> normalize(waveforms, norm="l1")
        array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
               [-0.1  ,  0.167, -0.25 ,  0.25 ],
               [ 0.   ,  0.   ,  0.   ,  0.25 ],
               [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
        >>> # l2-normalize the columns
        >>> normalize(waveforms, norm="l2")
         array([[-0.985,  0.943, -0.816,  0.5  ],
                [-0.123,  0.236, -0.408,  0.5  ],
                [ 0.   ,  0.   ,  0.   ,  0.5  ],
                [ 0.123,  0.236,  0.408,  0.5  ]])

    """

    # get the smallest normal as the threshold
    if np.issubdtype(waveforms.dtype, np.floating) or np.issubdtype(waveforms.dtype, np.complexfloating):
        dtype = waveforms.dtype
    else:
        dtype = np.float32

    threshold = np.finfo(dtype).tiny

    # perform norm on magnitude only
    mag = np.abs(waveforms).astype(np.float)

    if norm == "mean":
        mean = np.mean(mag, axis=axis, keepdims=True)
        return waveforms - mean

    elif norm == "mean_std":
        mean = np.mean(mag, axis=axis, keepdims=True)
        std = np.std(mag, axis=axis, keepdims=True)
        return (waveforms - mean) / (std + 1e-5)

    elif norm == "max":
        scale = np.max(mag, axis=axis, keepdims=True)

    elif norm == "min":
        scale = np.min(mag, axis=axis, keepdims=True)

    elif norm == "l0":
        scale = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif norm == "l1":
        scale = np.sum(mag, axis=axis, keepdims=True)

    elif norm == "l2":
        scale = np.sum(mag ** 2, axis=axis, keepdims=True) ** (1.0 / 2)

    else:
        raise TypeError("Unsupported norm type {}".format(repr(norm)))

    # indices where scale is below the threshold
    idx = scale < threshold
    Xnorm = np.empty_like(waveforms)
    # leave small indices un-normalized
    scale[idx] = 1.0
    Xnorm[:] = waveforms / scale

    return Xnorm


def unitarize(waveforms, lengths=None, amp_type="avg", eps=1e-14):
    """Normalizes a signal to unitary average or peak amplitude.

    Args:
        waveforms (np.ndarray): The waveforms to normalize. Shape should be `[batch, num_frames]` or
            `[batch, num_frames, channel]`.
        lengths (int or float): The lengths of the waveforms excluding the padding.
        amp_type (str): Whether one wants to normalize with respect to "avg" or "peak".
        eps (float): A small number to add to the denominator to prevent NaN.

    Returns:
        np.ndarray, unitarized level waveform.
    """
    assert amp_type in ["avg", "peak"]

    batch_added = False

    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = np.expand_dims(waveforms, 0)

    den = compute_amplitude(waveforms, lengths, amp_type) + eps
    if batch_added:
        waveforms = waveforms.squeeze(0)
    return waveforms / den


def rescale(waveforms, target_lvl, lengths=None, amp_type="avg", dB=False):
    """
    Performs signal rescaling to a target level.

    Args:
        waveforms (np.ndarray): The waveforms to be rescaled.
        target_lvl (float): Target level in dB or linear scale the waveforms to be rescaled to.
        lengths (int or None): The length of waveforms to be kept as rescale factor.
        amp_type (str): Whether one wants to rescale with maximum value or average. Options: ["avg", "max"].
        dB (bool): Whether target_lvl will be turned into dB scale. Options: [True, False].

    Returns:
        np.ndarray, the rescaled waveformes

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>>
        >>> waveforms = np.arange(10)
        >>> target_lvl = 2
        >>> rescaled_waves = rescale(waveforms,target_lvl=target_lvl,amp_type="avg")
        >>> compute_amplitude(rescaled_waves)
        2
    """

    assert amp_type in ["max", "avg"]
    assert dB in [True, False]

    batch_added = False

    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = np.expand_dims(waveforms, 0)

    waveforms = unitarize(waveforms, lengths=lengths, amp_type=amp_type)
    if dB:
        out = dB_to_amplitude(np.array(target_lvl), ref=1.0, power=0.5) * waveforms
    else:
        out = target_lvl * waveforms

    if batch_added:
        out = out.squeeze(0)

    return out


def stereo_to_mono(waveforms):
    """
    Transform stereo audios into mono audio by averaging different channels.

    Args:
        waveforms (np.ndarray): [shape=(n,2) or shape=(n,)] audio signals.

    Returns:
         np.ndarray, shape(n,) mono audio.

    Examples:
        >>> y = np.array([[1, 2], [0.5, 0.1]])
        >>> y = stereo_to_mono(y)
        >>> np.allclose(np.array([0.75, 1.05]), y)
        True
    """
    # Ensure Fortran contiguity.
    # data = np.asfortranarray(data)
    if isinstance(waveforms, ms.Tensor):
        import mindspore.numpy as np
    else:
        import numpy as np
    if waveforms.ndim > 1:
        waveforms = np.mean(waveforms, axis=-1)
    return waveforms


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


def trim(waveforms, top_db=60, reference=np.max, frame_length=2048, hop_length=512):
    """
    Trim an audio signal to keep concecutive non-silent segment.

    Args:
        waveforms (np.ndarray): The audio signal in shape (n,) or (n, n_channel).
        top_db (float): The threshold in decibels below `reference`. The audio segments below this threshold compared to
            `reference` will be considered as silence.
        reference (float, Callable): The reference amplitude. By default, `np.max` is used to serve as the reference
            amplitude.
        frame_length (int): The number of frames per analysis.
        hop_length (int): The number of frames between analysis.

    Returns:
        np.ndarray, the trimmed signal.
        np.ndarray, the index corresponding to the non-silent region: `wav_trimmed = waveforms[index[0]: index[1]]`
            (for mono) or `wav_trimmed = waveforms[index[0]: index[1], :]`.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveforms = np.array([0.01]*1000 + [0.6]*1000 + [-0.6]*1000)
        >>> wav_trimmed, index = trim(waveforms, top_db=10)
        >>> wav_trimmed.shape
        (2488,)
        >>> index[0]
        512
        >>> index[1]
        3000
    """
    mono_data = stereo_to_mono(waveforms)
    mono_data = np.pad(mono_data, int(frame_length // 2))
    x = frame(mono_data, frame_length=frame_length, hop_length=hop_length)
    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=0, keepdims=False)
    rms = np.sqrt(power) ** 2
    non_silent = amplitude_to_dB(rms, ref=reference, top_db=None) > -top_db

    edges = np.flatnonzero(np.diff(non_silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    index = []
    if non_silent[0]:
        index.append(0)
    else:
        index.append(edges[0][0])
    if non_silent[-1]:
        index.append(len(non_silent))
    else:
        index.append(edges[-1][-1])

    return waveforms[int(index[0] * hop_length):int(index[1] * hop_length)], np.array(index) * hop_length


def split(waveforms, top_db=60, reference=np.max, frame_length=2048, hop_length=512):
    """
    Split an audio signal into non-silent intervals.

    top_db (float): The threshold in decibels below `reference`. The audio segments below this threshold compared to
            `reference` will be considered as silence.
        reference (float, Callable): The reference amplitude. By default, `np.max` is used to serve as the reference
            amplitude.
        frame_length (int): The number of frames per analysis.
        hop_length (int): The number of frames between analysis.

    Returns:
        np.ndarray, shape=(m, 2), the index describing the start and end index of non-silent interval.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveforms = np.array([0.01]*2048 + [0.6]*2048 + [-0.01]*2048 + [0.5]*2048)
        >>> indices = split(waveforms, top_db=10)
        >>> indices.shape
        (2, 2)
        >>> indices
        array([[1536, 5120],
               [5632, 8192]])
    """
    mono_data = stereo_to_mono(waveforms)
    mono_data = np.pad(mono_data, int(frame_length // 2))
    x = frame(mono_data, frame_length=frame_length, hop_length=hop_length)
    # Calculate power
    power = np.mean(np.abs(x) ** 2, axis=0, keepdims=False)
    rms = np.sqrt(power) ** 2
    non_silent = amplitude_to_dB(rms, ref=reference, top_db=None) > -top_db

    edges = np.flatnonzero(np.diff(non_silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    # If the first frame had high energy, count it
    if non_silent[0]:
        edges.insert(0, [0])

    # Likewise for the last frame
    if non_silent[-1]:
        edges.append([len(non_silent)])

    # Convert from frames to samples
    edges = np.concatenate(edges) * hop_length

    # Clip to the signal duration
    edges = np.minimum(edges, waveforms.shape[-1])

    # Stack the results back as an ndarray
    return edges.reshape((-1, 2))


def reverberate(waveforms, rir_waveform, rescale_amp="avg"):
    """
    Reverberate a given signal with given a Room Impulse Response (RIR). It performs convolution between RIR and signal,
    but without changing the original amplitude of the signal.

    Args:
        waveforms (np.ndarray): The audio signal to reverberate.
        rir_waveform (np.ndarray): The Room Impulse Response signal.
        rescale_amp (str): Whether reverberated signal is rescaled (None) and with respect either to original signal
            "peak" amplitude or "avg" average amplitude. Options: [None, "avg", "peak"].

    Returns:
        np.ndarray, the reverberated signal.
    """

    orig_shape = waveforms.shape

    if len(waveforms.shape) > 3 or len(rir_waveform.shape) > 3:
        raise NotImplementedError

    # if inputs are mono tensors we reshape to 1, samples
    if len(waveforms.shape) == 1:
        waveforms = np.expand_dims(np.expand_dims(waveforms, 0), -1)
    elif len(waveforms.shape) == 2:
        waveforms = np.expand_dims(waveforms, -1)

    if len(rir_waveform.shape) == 1:  # convolve1d expects a 3d tensor !
        rir_waveform = np.expand_dims(np.expand_dims(rir_waveform, 0), -1)
    elif len(rir_waveform.shape) == 2:
        rir_waveform = np.expand_dims(rir_waveform, -1)

    # Compute the average amplitude of the clean
    orig_amplitude = compute_amplitude(
        waveforms, waveforms.shape[1], rescale_amp
    )

    # Compute index of the direct signal, so we can preserve alignment
    value_max = np.max(np.abs(rir_waveform), axis=1, keepdims=True)
    direct_index = np.argmax(np.abs(rir_waveform))

    # Making sure the max is always positive (if not, flip)
    # mask = torch.logical_and(rir_waveform == value_max,  rir_waveform < 0)
    # rir_waveform[mask] = -rir_waveform[mask]

    # Use FFT to compute convolution, because of long reverberation filter
    waveforms = convolve1d(
        waveforms=waveforms,
        kernel=rir_waveform,
        use_fft=True,
        rotation_index=direct_index,
    )

    if len(orig_shape) == 1:
        waveforms = np.squeeze(np.squeeze(waveforms, 0), -1)
        lengths = len(waveforms)
    if len(orig_shape) == 2:
        waveforms = np.squeeze(waveforms, -1)
        lengths = waveforms.shape[1]
    if len(orig_shape) == 3:
        lengths = waveforms.shape[1]

    # Rescale to the peak amplitude of the clean waveform
    waveforms = rescale(
        waveforms, orig_amplitude, lengths=lengths, amp_type=rescale_amp
    )
    return waveforms


def convolve1d(waveforms, kernel, padding=0, pad_type="constant", stride=1, groups=1, use_fft=False, rotation_index=0):
    """Use mindspore.conv1d to perform 1d padding and convolution.

    Args:
        waveforms (np.ndarray): The audio signal to perform convolution on.
        kernel (np.ndarray): The filter kernel to apply during convolution.
        padding (int, tuple): The padding size to apply at left side and right side.
        pad_type (str): The type of padding to use. Options: ["constant", "edge"].
        stride (int): The number of units to stride for the convolution operations. If `use_fft` is True, this will not
            have effects.
        groups (int): This option is passed to `conv1d` to split the input into groups for convolution. Input channels
            should be divisible by the number of groups.
        use_fft (bool): When `use_fft` is passed `True`, then compute the convolution in the spectral domain using
            complex multiply. This is more efficient on CPU when the size of the kernel is large (e.g. reverberation).
            WARNING: Without padding, circular convolution occurs. This makes little difference in the case of
            reverberation, but may make more difference with different kernels.
        rotation_index (int): This option only applies if `use_fft` is true. If so, the kernel is rolled by this amount
            before convolution to shift the output location.

    Returns:
        np.ndarray, the convolved waveform.
    """

    n_dim = len(waveforms.shape)
    if n_dim == 1:
        waveforms = np.expand_dims(np.expand_dims(waveforms, -1), 0)
    if len(kernel.shape) == 1:
        kernel = np.expand_dims(np.expand_dims(kernel, -1), 0)

    # batchify waveforms and kernels
    if n_dim == 2:
        waveforms = np.expand_dims(waveforms, -1)
    if n_dim == 2:
        kernel = np.expand_dims(kernel, -1)

    waveforms = np.transpose(waveforms, [0, 2, 1])  # make sure time last
    kernel = np.transpose(kernel, [0, 2, 1])  # make sure time last

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        waveforms = np.pad(waveforms, pad=padding, mode=pad_type)

    # This approach uses FFT, which is more efficient if the kernel is large
    if use_fft:

        # Pad kernel to same length as signal, ensuring correct alignment
        zero_length = waveforms.shape[-1] - kernel.shape[-1]

        # Handle case where signal is shorter
        if zero_length < 0:
            kernel = kernel[..., :zero_length]
            zero_length = 0

        # Perform rotation to ensure alignment
        zeros = np.zeros((kernel.shape[0], kernel.shape[1], zero_length))

        after_index = kernel[..., rotation_index:]
        before_index = kernel[..., :rotation_index]
        kernel = np.concatenate((after_index, zeros, before_index), axis=-1)

        result = np.fft.rfft(waveforms) * np.fft.rfft(kernel)
        convolved = np.fft.irfft(result, n=waveforms.shape[-1])

    else:
        # Todo: the implementation can be optimized here.
        conv1d = Conv1d(1, 1, kernel_size=kernel.shape[-1],
                        stride=stride, group=groups, padding=0, pad_mode='valid')
        weight = ms.Tensor(np.expand_dims(kernel, 0))
        weight.set_dtype(ms.float32)
        conv1d.weight.set_data(weight)
        data = ms.Tensor(waveforms).astype(ms.float32)
        convolved = conv1d(data).asnumpy()

    if n_dim == 1:  # meaning num_channel and batch dimension are expanded
        convolved = np.squeeze(np.squeeze(convolved, 1), 0)
        return convolved
    if n_dim == 2:  # meaning num_channel is expanded
        convolved = np.squeeze(convolved, 1)
        return convolved
    if n_dim == 3:
        return np.transpose(convolved, [0, 2, 1])  # transpose back to channel last
