import mindspore as ms
import mindspore.dataset.audio as msaudio
import numpy as np
from mindspore import Tensor, nn
from mindspore.dataset.audio.utils import BorderType, NormMode, WindowType, create_dct
from scipy.ndimage import median_filter

from .spectrum import amplitude_to_dB, istft, magphase, melspectrogram, stft

__all__ = [
    "spectral_centroid",
    "context_window",
    "compute_deltas",
    "fbank",
    "mfcc",
    "complex_norm",
    "angle",
    "harmonic",
]


def spectral_centroid(
    waveforms,
    sample_rate,
    n_fft=400,
    win_length=None,
    hop_length=None,
    pad=0,
    window="hann",
):
    """
    Create a spectral centroid from an audio signal.

    Args:
        waveforms (np.ndarray): Audio signal
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
        win_length (int): Window size (default=None, will use n_fft).
        hop_length (int): Length of hop between STFT windows (default=None,
        will use win_length // 2).
        pad (int): Two sided padding of signal (default=0).
        window (WindowType): Window function that is applied/multiplied to
        each frame/window,which can be
        WindowType.BARTLETT, WindowType.BLACKMAN, WindowType.HAMMING,
        WindowType.HANN or WindowType.KAISER
        (default=WindowType.HANN).

    Returns:
        Dimension `(..., time)`

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.features as features
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> spectralcentroid = features.spectral_centroid(waveform, sr)

    """

    win_length = win_length if win_length else n_fft
    hop_length = hop_length if hop_length else win_length // 2
    window = WindowType(window)
    spectralcentroid = msaudio.SpectralCentroid(
        sample_rate, n_fft, win_length, hop_length, pad, window
    )

    return spectralcentroid(waveforms)


def context_window(waveforms, left_frames=0, right_frames=0):
    """
    Create a context window from an audio signal to gather multiple time step
    in a single feature vector.
    Returns the array with the surrounding context.

    Args:
        waveforms(np.ndarray): Single-channel or multi-channel time-series
        audio signals with shape [freq, time],[batch, freq, time] or
        [batch, channel, freq, time].
        left_frames (int): Number of past frames to collect.
        left_frames (int): Number of future frames to collect.

    Returns:
        np.array: Aggregated feature vector by gathering the past and future
        time steps. The feature with shape [freq, time], [batch, freq, time]
        or [batch, channel, freq, time].

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.features as features
        >>> input_arr = np.random.randn(10, 101, 60).astype(dtype=np.float32)
        >>> contextwin = features.context_window(input_arr)
    """

    context_size = left_frames + right_frames + 1
    kernel_size = 2 * max(left_frames, right_frames) + 1
    shift = right_frames - left_frames
    max_frame = max(left_frames, right_frames)
    # Kernel definition
    kernel = np.eye(context_size, kernel_size, dtype=np.float32)
    if shift > 0:
        kernel = np.roll(kernel, shift, 1)
    first_call = True

    input_shape = waveforms.shape
    # Considering multi-channel case
    if len(input_shape) == 2:
        x = np.expand_dims(waveforms, 0)
    elif len(input_shape) == 3:
        x = waveforms
    elif len(input_shape) == 4:
        x = waveforms.transpose((0, 2, 3, 1))
    else:
        raise TypeError(
            "Input dimension must be 2, 3 or 4, but got {}".format(len(input_shape))
        )

    if first_call:
        first_call = False
        tile = np.tile(kernel, (x.shape[1], 1, 1))
        tile = tile.reshape((x.shape[1] * context_size, kernel_size,))
        kernel = np.expand_dims(tile, 1)

    x_shape = x.shape
    if len(x_shape) == 4:
        x = x.reshape((x_shape[0] * x_shape[2], x_shape[1], x_shape[3]))

    # Computing context using the estimated kernel
    in_channel, out_channel = x_shape[1], kernel.shape[0]
    conv = nn.Conv1d(
        in_channel,
        out_channel,
        kernel_size,
        padding=max_frame,
        pad_mode="pad",
        group=x.shape[1],
        weight_init=Tensor(kernel),
    )
    x_tensor = Tensor(x, ms.float32)
    context = conv(x_tensor)
    # Retrieving the original dimensionality for multi-channel case
    if len(x_shape) == 4:
        context = context.reshape(
            (x_shape[0], context.shape[1], x_shape[2], context.shape[-1])
        )
        context = context.transpose((0, 3, 1, 2))

    if len(x_shape) == 2:
        context = np.squeeze(context, 0)

    return context.asnumpy()


def compute_deltas(specgram, win_length=5, pad_mode="edge"):
    """
    Compute delta coefficients of a spectrogram.

    Args:
        specgram(np.ndarray): audio signals of dimension'(..., freq, time)'
        win_length (int): The window length used for computing delta, must be
        no less than 3 (default=5).
        pad_mode (str): Mode parameter passed to padding, which can be
        'constant', 'edge', 'reflect', 'symmetric'
            (default='edge').

            - 'constant', means it fills the border with constant values.

            - 'edge', means it pads with the last value on the edge.

            - 'reflect', means it reflects the values on the edge omitting the
            last value of edge.

            - 'symmetric', means it reflects the values on the edge repeating
            the last value of edge.
    Returns:
        deltas

    Examples:
        >>> import numpy as np
        >>> import mindaudio.data.features as features
        >>> specgram = np.random.random([1, 400 // 2 + 1, 1000])
        >>> deltas = features.compute_deltas(waveforms, win_length=7, \
        pad_mode="edge")
    """

    pad_mode = BorderType(pad_mode)
    compute_deltas_ms = msaudio.ComputeDeltas(win_length, pad_mode)

    return compute_deltas_ms(specgram)


def fbank(
    waveforms,
    deltas=False,
    context=False,
    n_mels=40,
    n_fft=400,
    sample_rate=16000,
    f_min=0.0,
    f_max=None,
    left_frames=5,
    right_frames=5,
    win_length=None,
    hop_length=None,
    window="hann",
):
    """
    Generate filter bank features.

    Args:
        waveforms(np.ndarray): A batch of audio signals with shape [time],
        [batch, time] or [batch, channel, time].
        deltas (bool): Whether or not to append derivatives and second
        derivatives to the features (default: False).
        context (bool): Whether or not to append forward and backward contexts
        to the features (default: False).
        n_mels (int): Number of Mel filters (default: 40).
        n_fft (int): Number of samples used in each stft (default: 400).
        sample_rate (int): Sampling rate for the input waveforms
        (default: 160000).
        f_min (float): Minimum frequency (default=0).
        f_max (float): Maximum frequency (default=None, will be set to
        sample_rate // 2).
        left_frames (int): Number of frames  left context to collect
        (default: 5).
        right_frames (int): Number of past frames to collect (default: 5).
        win_length (int): Window size (default=None, will use n_fft).
        hop_length (int): Length of hop between STFT windows
        (default=None, will use win_length // 2).
        window (str): Window function that is applied/multiplied to each
        frame/window,which can be 'bartlett',
        'blackman', 'hamming', 'hann' or 'kaiser' (default='hann').
        Currently kaiser window is not supported on macOS.

    Returns:
        np.ndarray: Mel-frequency cepstrum coefficients with shape
        [freq, time], [batch, freq, time] or [batch, channel, freq, time].

    Example:
        >>> import numpy as np
        >>> import mindaudio.data.features as features
        >>> inputs = np.random.random([10, 16000])
        >>> feats = features.fbanks(inputs)
        >>> feats.shape
        (10, 40, 101)
    """

    melspcgram = melspectrogram(
        waveforms,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
    )
    fbanks = amplitude_to_dB(wavform=melspcgram, stype="power", ref=1.0, top_db=80.0)
    if deltas:
        delta1 = compute_deltas(fbanks)
        delta2 = compute_deltas(delta1)
        fbanks = np.concatenate((fbanks, delta1, delta2), axis=-2)
    if context:
        fbanks = context_window(fbanks, left_frames, right_frames)
    return fbanks


def mfcc(
    waveforms,
    deltas=True,
    context=True,
    n_mels=23,
    n_mfcc=20,
    n_fft=400,
    sample_rate=16000,
    f_min=0.0,
    f_max=None,
    left_frames=5,
    right_frames=5,
    win_length=None,
    hop_length=None,
    norm="ortho",
    log_mels=False,
):
    """Generate Mel-frequency cepstrum coefficients (MFCC) features from input
    audio signal.

    Args:
        waveforms(np.ndarray): A batch of audio signals with shape [time],
        [batch, time] or [batch, channel, time].
        deltas (bool): Whether or not to append derivatives and second
        derivatives to the features (default: False).
        context (bool): Whether or not to append forward and backward contexts
        to the features (default: False).
        n_mels (int): Number of Mel filters (default: 23).
        n_mfcc (int): Number of Mel-frequency cepstrum coefficients
        (default: 20).
        n_fft (int): Number of samples used in each stft (default: 400).
        sample_rate (int): Sampling rate for the input waveforms
        (default: 160000).
        f_min (float, optional): Minimum frequency (default=0).
        f_max (float, optional): Maximum frequency (default=None, will be set
        to sample_rate // 2).
        left_frames (int): Number of frames  left context to collect
        (default: 5).
        right_frames (int): Number of past frames to collect (default: 5).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows
        (default=None, will use win_length // 2).
        norm (str, optional): Normalization mode, can be "none" or "orhto"
        (default="none").
        log_mels (bool, optional): Whether to use log-mel spectrograms instead
        of db-scaled (default=False).

    Returns:
        np.ndarray: Mel-frequency cepstrum coefficients with shape
        [freq, time], [batch, freq, time] or [batch, channel, freq, time].

    Example:
        >>> import numpy as np
        >>> inputs = np.random.random([10, 16000])
        >>> feats = features.mfcc(inputs)
        >>> feats.shape
        (10, 660, 101)
    """
    norm = NormMode(norm)

    if n_mfcc > n_mels:
        raise ValueError(
            "The number of MFCC coefficients must be no more than # mel bins."
        )
    dct = create_dct(n_mfcc=n_mfcc, n_mels=n_mels, norm=norm)

    melspec = melspectrogram(
        waveforms,
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        win_length=win_length,
        hop_length=hop_length,
    )
    if log_mels:
        melspec = np.log(melspec + 1e-6)
    else:
        melspec = amplitude_to_dB(melspec, stype="power", ref=1.0, top_db=80.0)
    melspecgram_shape = melspec.shape
    # Considering multi-channel case
    # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_mfcc, time)
    if len(melspecgram_shape) == 2:
        mfccs = np.matmul(melspec.transpose((1, 0)), dct).transpose((1, 0))
    elif len(melspecgram_shape) == 3:
        mfccs = np.matmul(melspec.transpose((0, 2, 1)), dct).transpose((0, 2, 1))
    elif len(melspecgram_shape) == 4:
        mfccs = np.matmul(melspec.transpose((0, 1, 3, 2)), dct).transpose((0, 1, 3, 2))
    else:
        raise TypeError(
            "Unsupported MelSpectrogram shape {}".format(len(melspecgram_shape))
        )

    if deltas:
        delta1 = compute_deltas(mfccs)
        delta2 = compute_deltas(delta1)
        mfccs = np.concatenate((mfccs, delta1, delta2), axis=-2)
    if context:
        mfccs = context_window(mfccs, left_frames, right_frames)
    return mfccs


def complex_norm(waveforms, power=1.0):
    """
    Compute the norm of complex number sequence.

    Note:
        The dimension of the audio waveform to be processed needs to be
        (..., complex=2).
        The first dimension represents the real part while the second
        represents the imaginary.

    Args:
        waveforms (np.ndarray): Array shape of (..., 2).
        power (float): Power of the norm, which must be non-negative.
        Default: 1.0.

    Returns:
            np.ndarray, norm of the input signal.

    Raises:
        TypeError: If `power` is not of type float.
        ValueError: If `power` is a negative number.
        RuntimeError: If input tensor is not in shape of <..., complex=2>.

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.features as features
        >>> import mindaudio.data.spectrum as spectrum
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> inputs_arr = spectrum.stft(waveform, return_complex=False)
        >>> norm = features.complex_norm(inputs_arr)
    """

    complex_norm = msaudio.ComplexNorm(power)

    return complex_norm(waveforms)


def angle(x):
    """
    Calculate the angle of the complex number sequence of shape (..., 2).
    The first dimension represents the real part while the second represents
    the imaginary.

    Args:
        x(np.ndarray): Complex number to compute

    Returns:
        np.ndarray, angle of the input signal.

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.features as features
        >>> import mindaudio.data.spectrum as spectrum
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> inputs_arr = spectrum.stft(waveform, return_complex=False)
        >>> angle = features.angle(inputs_arr)

    """
    angle_ms = msaudio.Angle()
    return angle_ms(x)


def soft_mask(x_input, x_ref, *, power=1, split_zeros=False):
    if np.any(x_input < 0) or np.any(x_ref < 0):
        raise TypeError("x_input and x_ref must be non-negative")

    if x_input.shape != x_ref.shape:
        raise TypeError("x_input and x_ref shape mismatch.")

    if power <= 0:
        raise TypeError("power must be strictly positive.")

    input_type = x_input.dtype
    if not np.issubdtype(input_type, np.floating):
        input_type = np.float32

    z = np.maximum(x_input, x_ref).astype(input_type)
    bad_idx = z < np.finfo(input_type).tiny
    z[bad_idx] = 1

    if not np.isfinite(power):
        mask = x_input > x_ref
    else:
        mask = (x_input / z) ** power
        ref_mask = (x_ref / z) ** power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]

        if not split_zeros:
            mask[bad_idx] = 0.0
        else:
            mask[bad_idx] = 0.5

    return mask


def hpss(spectrogram, *, kernel_size=31, power=2.0, mask=False, margin=1.0):
    # pylint: disable=C,R,W,E,F
    if not np.iscomplexobj(spectrogram):
        phase = 1
    else:
        spectrogram, phase = magphase(spectrogram, power=1)

    if not np.isscalar(margin):
        margin_harmonic = margin[0]
        margin_perc = margin[1]
    else:
        margin_harmonic = margin
        margin_perc = margin

    if not np.isscalar(kernel_size):
        win_harmonic = kernel_size[0]
        win_perc = kernel_size[1]
    else:
        win_harmonic = kernel_size
        win_perc = kernel_size

    # margin minimum is 1.0
    if margin_harmonic < 1 or margin_perc < 1:
        raise TypeError(
            "Margins must be >= 1.0. " "A typical range is between 1 and 10."
        )

    # shape for kernels
    perc_shape = [1 for _ in spectrogram.shape]
    perc_shape[-2] = win_perc

    harmonic_shape = [1 for _ in spectrogram.shape]
    harmonic_shape[-1] = win_harmonic

    # Compute median filters. Pre-allocation here preserves memory layout.
    harm = np.empty_like(spectrogram)
    harm[:] = median_filter(spectrogram, size=harmonic_shape, mode="reflect")

    perc = np.empty_like(spectrogram)
    perc[:] = median_filter(spectrogram, size=perc_shape, mode="reflect")

    split_zeros = margin_harmonic == 1 and margin_perc == 1

    mask_harmonic = soft_mask(
        harm, perc * margin_harmonic, power=power, split_zeros=split_zeros
    )

    mask_perc = soft_mask(
        perc, harm * margin_perc, power=power, split_zeros=split_zeros
    )

    if mask:
        return mask_harmonic, mask_perc

    return (
        (spectrogram * mask_harmonic) * phase,
        (spectrogram * mask_perc) * phase,
    )


def harmonic(y_input, **kwargs):
    """
    Extract harmonic elements from an audio time-series.

    Args:
        y_input(np.ndarray): A batch of data in shape (,n) or (n_channel,n).
        **kwargs : additional keyword arguments.

    Returns:
        np.ndarray, the waveform after harmonic,A batch of data in shape (,n)
        or (n_channel,n).

    Examples:
        >>> waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> harm = features.harmonic(waveform)
        >>> # Use a margin > 1.0 for greater harmonic separation
        >>> harm = features.harmonic(waveform, margin=3.0)
    """
    # STFT
    y_stft = stft(y_input, n_fft=2048, pad_mode="constant")

    # Remove percussive
    stft_harm = hpss(y_stft, **kwargs)[0]

    # Inverse STFT
    y_harm = istft(stft_harm, length=y_input.shape[-1])

    return y_harm
