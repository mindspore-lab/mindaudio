import math

import mindspore as ms
import mindspore.dataset.audio as msaudio
import numpy as np
import scipy
from mindspore import Parameter, Tensor, ops

from .spectrum import amplitude_to_dB, compute_amplitude, dB_to_amplitude, frame

__all__ = [
    "normalize",
    "unitarize",
    "resample",
    "rescale",
    "stereo_to_mono",
    "trim",
    "split",
    "sliding_window_cmn",
    "invert_channels",
    "loop",
    "clip",
    "insert_in_background",
    "overlap_and_add",
]


def normalize(waveforms, norm="max", axis=0):
    """Normalize an array along a specified axis.

    Args:
        waveforms (np.ndarray): An audio signal to normalize
        norm (Union['min', 'max', 'mean', 'mean_std', 'l0', 'l1', 'l2']): Normalization type,
        here "max" means l-infinity norm, "mean_std" refers to the mean standard deviation norm
        axis (int): Axis along which to compute the norm

    Raises:
        TypeError: If the normalization type is not supported

    Returns:
        np.ndarray, normalized array.

    Examples:
        >>> waveforms = np.vander(np.arange(-2, 2))
        >>> normalize(waveforms, axis=1)
        >>> normalize(waveforms, norm="min")
        >>> normalize(waveforms, norm="l0")
    """

    # get the smallest normal as the threshold
    if np.issubdtype(waveforms.dtype, np.floating) or np.issubdtype(
        waveforms.dtype, np.complexfloating
    ):
        dtype = waveforms.dtype
    else:
        dtype = np.float32

    threshold = np.finfo(dtype).tiny

    # perform norm on magnitude only
    mag = np.abs(waveforms).astype(float)

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

    x_norm = np.empty_like(waveforms)
    idx = scale < threshold
    scale[idx] = 1.0
    x_norm[:] = waveforms / scale

    return x_norm


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


def resample(
    waveform,
    orig_freq=16000,
    new_freq=16000,
    res_type="fft",
    lowpass_filter_width=6,
    rolloff=0.99,
    beta=None,
):
    """
    Resample a signal from one frequency to another. A resample method can be given.

    Args:
        waveform (np.ndarray): The waveforms to normalize. Shape should be `[batch, num_frames]` or
            `[batch, num_frames, channel]`.
        orig_freq (float): The original frequency of the signal, which must be positive (default=16000).
        new_freq (float): The desired frequency, which must be positive (default=16000).
        res_type (str): The resample method, which can be "fft","scipy","minddata".
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
    if orig_freq == new_freq:
        return waveform

    ratio = float(new_freq) / orig_freq

    n_samples = int(np.ceil(waveform.shape[-1] * ratio))

    if res_type in ("scipy", "fft"):
        y_hat = scipy.signal.resample(waveform, n_samples, axis=-1)
        return np.asarray(y_hat, dtype=waveform.dtype)

    else:
        resample_function = msaudio.Resample(
            orig_freq=orig_freq,
            new_freq=new_freq,
            lowpass_filter_width=lowpass_filter_width,
            rolloff=rolloff,
            beta=beta,
        )
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
        out_waveforms = (
            dB_to_amplitude(np.array(target_lvl), ref=1.0, power=0.5) * waveforms
        )
    else:
        out_waveforms = target_lvl * waveforms

    if batch_added:
        out_waveforms = out_waveforms.squeeze(0)

    return out_waveforms


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
        top_db (float): The threshold in decibels below `reference`. The audio segments below this threshold
            compared to `reference` will be considered as silence.
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

    return (
        waveforms[int(index[0] * hop_length) : int(index[1] * hop_length)],
        np.array(index) * hop_length,
    )


def split(waveforms, top_db=60, reference=np.max, frame_length=2048, hop_length=512):
    """
    Split an audio signal into non-silent intervals.

    Args:
        waveforms (np.ndarray): The audio signal in shape (n,) or (n, n_channel).
        top_db (float): The threshold in decibels below `reference`. The audio segments below this threshold
            compared to `reference` will be considered as silence.
        reference (float, Callable): The reference amplitude. By default, `np.max` is used to serve as the
            reference amplitude.
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


def sliding_window_cmn(
    x, cmn_window=600, min_cmn_window=100, center=False, norm_vars=False
):
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
    sliding_window_cmn_ms = msaudio.SlidingWindowCmn(
        cmn_window, min_cmn_window, center, norm_vars
    )
    return sliding_window_cmn_ms(x)


def invert_channels(waveform):
    """
    Inverts channels of the audio signal. If the audio has only one channel, no change is applied,
    Otherwise, it inverts the order of the channels.
    eg: for 4 channels, it returns channels in order [3, 2, 1, 0].

    Args:
        waveform(np.ndarray): A batch of data in shape (n,) or (n, n_channel).

    Returns:
        np.ndarray, the waveform after invert channels.

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.processing as processing
        >>> waveform = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        >>> out_waveform = processing.invert_channels(waveform)
    """
    if waveform.ndim > 1:
        col = waveform.shape[1] - 1
        waveform[:, [0, col]] = waveform[:, [col, 0]]

    return waveform


def loop(waveform, times):
    """
    Loops the audio signal times.

    Args:
        waveform(np.ndarray): A batch of data in shape (n,) or (n, n_channel).
        times: the number of times the audio will be looped.

    Returns:
        np.ndarray, the waveform after loop n times.

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.processing as processing
        >>> waveform = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        >>> times = 3
        >>> out_waveform = processing.loop(waveform, times)
    """
    if times > 1:
        backup = waveform
        while times > 1:
            waveform = np.append(waveform, backup, axis=0)
            times -= 1

    return waveform


def clip(waveform, offset_factor, duration_factor):
    """
    Clips the audio using the specified offset and duration factors.

    Args:
        waveform(np.ndarray): A batch of data in shape (n,) or (n, n_channel).
        offset_factor: start point of the crop relative to the audio duration.
        duration_factor: the length of the crop relative to the audio duration.

    Returns:
        np.ndarray, the waveform after clip.

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.processing as processing
        >>> waveform = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        >>>                 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        >>> offset_factor = 0.1
        >>> duration_factor = 0.3
        >>> out_waveform = processing.clip(waveform, offset_factor, duration_factor)
    """
    if offset_factor + duration_factor < 0.0 or offset_factor + duration_factor > 1.0:
        print("Combination of offset and duration factors exceed audio length.")
        return waveform

    num_samples = waveform.shape[0]
    start = int(offset_factor * num_samples)
    end = int((offset_factor + duration_factor) * num_samples)
    out_waveform = waveform[start:end, ...]
    return out_waveform


def insert_in_background(waveform, offset_factor, background_audio):
    """
    Inserts audio signal into a background clip in a non-overlapping manner.

    Args:
        waveform(np.ndarray): A batch of data in shape (n,) or (n, n_channel).
        offset_factor: insert point relative to the background duration.
        background_audio(np.ndarray): A batch of data in shape (n,) or (n, n_channel),If set to `None`,
            the background audio will be white noise, with the same duration as the audio.

    Returns:
        np.ndarray, the waveform after insert background audio.

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.processing as processing
        >>> waveform = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        >>>                      [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]).T
        >>> offset_factor = 0.2
        >>> background_audio = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        >>>                             [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]]).T
        >>> out_waveform = processing.insert_in_background(waveform, offset_factor, background_audio)
    """
    if offset_factor < 0.0 or offset_factor > 1.0:
        print("Offset factor number exceed range [0, 1].")
        return waveform

    if background_audio is None:
        random_generator = np.random.mtrand._rand
        background_audio = random_generator.standard_normal(waveform.shape)
    else:
        num_channels = 1 if waveform.ndim == 1 else waveform.shape[1]
        bg_num_channels = 1 if background_audio.ndim == 1 else background_audio.shape[1]
        if bg_num_channels != num_channels:
            background_audio = stereo_to_mono(background_audio)
            if num_channels > 1:
                background_audio = np.expand_dims(background_audio, axis=1)
                background_audio = np.tile(background_audio, (1, num_channels))

    num_samples_bg = background_audio.shape[0]
    offset = int(offset_factor * num_samples_bg)
    if num_channels > 1:
        out_waveform = np.vstack(
            [background_audio[:offset, ...], waveform, background_audio[:offset, ...]]
        )
    else:
        out_waveform = np.hstack(
            [background_audio[..., :offset], waveform, background_audio[..., :offset]]
        )

    return out_waveform


def overlap_and_add(signal, frame_step):
    """
    Taken from https://github.com/kaituoxu/Conv-TasNet/blob/master/src/utils.py
    To factor code for mindspore

    Args:
        signal(mindspore.tensor): Shape of [..., frames, frame_length]. All dimensions may be unknown,
            and rank must be at least 2.
        frame_step(int): An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        overlapped(mindspore.tensor): With shape [..., output_size] containing the overlap-added frames
            of signal's inner-most two dimensions. output_size = (frames - 1) * frame_step + frame_length
    Based on
    https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py

    Example:
        >>> import mindspore as ms
        >>> signal = ms.Tensor(np.random.randn(5, 20), ms.float32)
        >>> overlapped = overlap_and_add(signal, 20)
        >>> overlapped.shape
    """

    outer_dimensions = signal.shape[:-2]
    frames, frame_length = signal.shape[-2:]

    # return the greatest common divisor of frame_length and frame_step
    cal_frame_length = math.gcd(frame_length, frame_step)
    cal_frame_step = frame_step // cal_frame_length
    subframes_per_frame = frame_length // cal_frame_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // cal_frame_length

    frame = np.lib.stride_tricks.sliding_window_view(
        np.arange(0, output_subframes), subframes_per_frame
    )[::cal_frame_step, :]
    frame = Tensor(frame.reshape(-1), ms.int32)

    new_zeros = ops.Zeros()
    result = new_zeros(
        (*outer_dimensions, output_subframes, cal_frame_length), ms.float32
    )
    # result.index_add_(-2, frame, subframe_signal)
    subframe_signal = signal.view(*outer_dimensions, -1, cal_frame_length)
    result = ops.index_add(Parameter(result), frame, subframe_signal, 1)
    result = result.view(*outer_dimensions, -1)
    return result
