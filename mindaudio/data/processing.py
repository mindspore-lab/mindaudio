import numpy as np
import mindspore as ms
from mindspore.dataset.audio import ResampleMethod
import mindspore.dataset.audio as msaudio
from .spectrum import dB_to_amplitude, amplitude_to_dB, compute_amplitude


__all__ = [
    'normalize',
    'unitarize',
    'resample',
    'rescale',
    'stereo_to_mono',
    'trim',
    'split',
]


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
        >>> waveforms = np.vander(np.arange(-2, 2))
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
    """
    Normalizes a signal to unitary average or peak amplitude.

    Args:
        waveforms (np.ndarray): The waveforms to normalize. Shape should be `[batch, num_frames]` or
            `[batch, num_frames, channel]`.
        lengths (int or float): The lengths of the waveforms excluding the padding.
        amp_type (str): Whether one wants to normalize with respect to "avg" or "peak".
        eps (float): A small number to add to the denominator to prevent NaN.

    Returns:
        np.ndarray, unitarized level waveform.

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.processing as processing
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> waveforms = processing.unitarize(waveform)
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


def resample(waveform, orig_freq=16000, new_freq=16000, resample_method=ResampleMethod.SINC_INTERPOLATION,
             lowpass_filter_width=6, rolloff=0.99, beta=None):
    """
    Resample a signal from one frequency to another. A resample method can be given.

    Args:
        orig_freq (float): The original frequency of the signal, which must be positive (default=16000).
        new_freq (float): The desired frequency, which must be positive (default=16000).
        resample_method (ResampleMethod, optional): The resample method, which can be
            ResampleMethod.SINC_INTERPOLATION and ResampleMethod.KAISER_WINDOW
            (default=ResampleMethod.SINC_INTERPOLATION).
        lowpass_filter_width (int): Controls the shaperness of the filter, more means sharper but less
            efficient, which must be positive (default=6).
        rolloff (float): The roll-off frequency of the filter, as a fraction of the Nyquist. Lower values
            reduce anti-aliasing, but also reduce some of the highest frequencies, range: (0, 1] (default=0.99).
        beta (float): The shape parameter used for kaiser window (default=None, will use 14.769656459379492).

    Returns:
        np.ndarray, unitarized level waveform.

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.processing as processing
        >>> waveform = np.random.random([1, 441000])
        >>> y_8k = processing.resample(waveform, orig_freq=44100, new_freq=16000)
        >>> print(waveform.shape)
        >>> print(y_8k.shape)
    """
    resample_function = msaudio.Resample(orig_freq=orig_freq, new_freq=new_freq, resample_method=resample_method,
             lowpass_filter_width=lowpass_filter_width, rolloff=rolloff, beta=beta)
    return resample_function(waveform)


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
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.processing as processing
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> target_lvl = 2
        >>> rescaled_waves = processing.rescale(waveform, target_lvl=target_lvl, amp_type="avg")
        >>> apm = spectrum.compute_amplitude(rescaled_waves)
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
        >>> import mindaudio.data.processing as processing
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
        >>> import mindaudio.data.processing as processing
        >>> waveforms = np.array([0.01]*1000 + [0.6]*1000 + [-0.6]*1000)
        >>> wav_trimmed, index = processing.trim(waveforms, top_db=10)
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

    Args:
        waveforms (np.ndarray): The audio signal in shape (n,) or (n, n_channel).
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
        >>> import mindaudio.data.processing as processing
        >>> waveforms = np.array([0.01]*2048 + [0.6]*2048 + [-0.01]*2048 + [0.5]*2048)
        >>> indices = processing.split(waveforms, top_db=10)
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


def sliding_window_cmn(x, cmn_window=600, min_cmn_window=100, center=False, norm_vars=False):
    """
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    Args:
        x (np.ndarray): A batch of data in shape (n,) or (n, n_channel).
        cmn_window (int): Window in frames for running average CMN computation (default=600).
        min_cmn_window (int): Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center is False, ignored if center is True (default=100).
        center (bool): If True, use a window centered on the current frame. If False, window is
            to the left. (default=False).
        norm_vars (bool): If True, normalize variance to one. (default=False).

    Returns:
        np.ndarray, the data after CMN

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.processing as processing
        >>> waveform = np.random.random([1, 20, 10])
        >>> after_CMN = processing.sliding_window_cmn(waveform, 500, 200)
    """
    sliding_window_cmn_ms = msaudio.SlidingWindowCmn(cmn_window, min_cmn_window, center, norm_vars)
    return sliding_window_cmn_ms(x)
