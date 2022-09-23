import numpy as np

from mindspore.dataset.audio.transforms import AudioTensorOperation
from mindspore.dataset.audio.utils import BorderType, WindowType, ScaleType, NormType, MelType
import mindspore.dataset.audio.transforms as transforms
from .functional import stft, istft, stereo_to_mono, rescale, trim, split, amplitude_to_dB, dB_to_amplitude


__all__ = [
    'STFT',
    'ISTFT',
    'Rescale',
    'StereoToMono',
    'Spectrogram',
    'MelSpectrogram',
    'AmplitudeToDB',
    'DBToAmplitude',
    'MagPhase',
    'ComputeDeltas',
    'Trim',
    'Split',
    'ComplexNorm',
    'Angle',
    'FrequencyMasking',
    'TimeMasking',
    'SlidingWindowCmn',
    'MelScale',
    'PreEmphasis'
]


class STFT(AudioTensorOperation):
    """
    Short-time Fourier transform (STFT).

    STFT represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short
    overlapping windows.

    Args:
        n_fft (int): Number of fft point of the STFT. It defines the frequency resolution (n_fft should be <= than
            win_length). The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``. In any case, we recommend
            setting ``n_fft`` to a power of two for optimizing the speed of the fast Fourier transform (FFT) algorithm.
        win_length (int): Number of frames the sliding window used to compute the STFT. Given the sample rate of the
            audio, the time duration for the windowed signal can be obtained as:
            :math:`duration (ms) = \frac{win_length*1000}{sample_rate}`. Usually, the time duration can be set to
            :math: `~30ms`. If None, win_length will be set to the same as n_fft.
        hop_length (int): Number of frames for the hop of the sliding window used to compute the STFT. If None,
            hop_length will be set to 1/4*n_fft.
        window (str, Callable): Window function specified for STFT. This function should take an integer (number of
            samples) and outputs an array to be multiplied with each window before fft.
        center (bool): If True (default), the input will be padded on both sides so that the t-th frame is centered at
            time t×hop_length. Otherwise, the t-th frame begins at time t×hop_length.
        pad_mode (str): Padding mode. Options: ["center", "reflect", "constant"]. Default: "reflect".
        return_complex (bool): Whether to return complex array or a real array for the real and imaginary components.
    """

    def __init__(self, n_fft=512, win_length=None, hop_length=None, window="hann", center=True,
                 pad_mode="reflect", return_complex=True):
        super(STFT, self).__init__()

        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.return_complex = return_complex

    def __call__(self, waveforms):
        """
        Execute the STFT to get the transformed matrix.

        Args:
            waveforms (np.ndarray): 1D or 2D array representing the time-serie audio signal.

        Returns:
            np.ndarray, STFT of waveform.

        Examples:
            >>> import numpy as np
            >>>
            >>> stft = STFT(n_fft=512)
            >>> data = np.arange(1024)/1024
            >>> stft(data).shape
            (257, 9)
        """

        stft_mat = stft(waveforms, self.n_fft, self.win_length, self.hop_length, self.window, self.center,
                        self.pad_mode, self.return_complex)
        return stft_mat


class ISTFT(AudioTensorOperation):
    r"""
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram ``stft_matrix`` to time-series ``data`` by minimizing the mean squared error
    between ``stft_matrix`` and STFT of ``y`` as described in [#]_ up to Section 2 (reconstruction from MSTFT). In
    general, window function, hop length and other parameters should be same as in stft, which mostly leads to perfect
    reconstruction of a signal from unmodified ``stft_matrix``.

    Args:
        n_fft (int): Number of fft point of the STFT. It defines the frequency resolution (n_fft should be <= than
            win_len * (sample_rate/1000)). The number of rows in the STFT matrix ``D`` is ``(1 + n_fft/2)``. In any
            case, we recommend setting ``n_fft`` to a power of two for optimizing the speed of the fast Fourier
            transform (FFT) algorithm.
        win_length (int): Number of frames the sliding window used to compute the STFT. Given the sample rate of the
            audio, the time duration for the windowed signal can be obtained as:
            :math:`duration (ms) = \frac{win_length*1000}{sample_rate}`. Usually, the time duration can be set to
            :math: `~30ms`. If None, win_length will be set to the same as n_fft.
        hop_length (int): Number of frames for the hop of the sliding window used to compute the STFT. If None,
            hop_length will be set to 1/4*n_fft.
        window (str, Callable): Window function specified for STFT. This function should take an integer (number of
            samples) and outputs an array to be multiplied with each window before fft.
        center (bool): If True (default), the input will be padded on both sides so that the t-th frame is centered at
            time t*hop_length. Otherwise, the t-th frame begins at time t*hop_length.
    """

    def __init__(self, n_fft=None, win_length=None, hop_length=None, window="hann", center=True):
        super(ISTFT, self).__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = window
        self.center = center

    def __call__(self, stft_matrix):
        """
        Execute the ISTFT to get the inversed transformed audio signal.

        Args:
            stft_matrix (np.ndarray): The STFT spectrogram.

        Returns:
            Tensor, the time domain signal.

        Examples:
            >>> import numpy as np
            >>>
            >>> stft = STFT(n_fft=512)
            >>> data = np.arange(1024)/1024
            >>> stft_matrix = stft(data)
            >>> istft = ISTFT()
            >>> output = istft(stft_matrix)
            >>> output.shape
            (1024,)
        """
        return istft(stft_matrix, self.n_fft, self.win_length, self.hop_length, self.window, self.center)


class Rescale(AudioTensorOperation):
    """
    Rescale the waveforms into the target level of amplitude/dB.

    Args:
        target_lvl (float): Target level in dB or linear scale the waveforms to be rescaled to.
        amp_type (str): Whether one wants to rescale with maximum value or average. Options: ["avg", "max"].
        dB (bool): Whether target_lvl will be turned into dB scale. Options: [True, False].
    """

    def __init__(self, target_lvl, amp_type="avg", dB=False):
        super(Rescale, self).__init__()
        self.target_lvl = target_lvl
        self.amp_type = amp_type
        self.dB = dB

    def __call__(self, waveforms):
        """
        Execute the rescale operation.

        Args:
            waveforms (Tensor):

        Returns:
            Tensor, the rescaled waveformes.

        Examples:
            >>> import numpy as np
            >>>
            >>> waveforms = np.arange(10)
            >>> target_lvl = 2
            >>> rescale = Rescale(target_lvl, "avg")
            >>> rescaled_waves = rescale(waveforms)
            >>> compute_amplitude(rescaled_waves)
            2
        """
        return rescale(waveforms, target_lvl=self.target_lvl, amp_type=self.amp_type, dB=self.dB)


class StereoToMono(AudioTensorOperation):

    def __call__(self, waveforms):
        """
        Converts stereo waveforms to mono waveforms.

        Args:
            waveforms (np.ndarray): [shape=(n,2) or shape=(n,)] Audio signal.

        Returns:
            np.ndarray, shape(n,) mono audio.

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>>
            >>> y = np.array([[1, 2], [0.5, 0.1]])
            >>> stereo_to_mono = StereoToMono()
            >>> y = stereo_to_mono(y)
            >>> np.allclose(np.array([0.75, 1.05]), y)
            True
        """
        return stereo_to_mono(waveforms)


class Spectrogram(AudioTensorOperation):
    """
    Create a spectrogram from an audio signal.

    Args:
        n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use win_length // 2).
        pad (int): Two sided padding of signal (default=0).
        window (str, optional): Window function that is applied/multiplied to each frame/window,
            which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann'). Currently kaiser
            window is not supported on macOS.
        power (float, optional): Exponent for the magnitude spectrogram, which must be greater
            than or equal to 0, e.g., 1 for energy, 2 for power, etc. (default=2.0).
        normalized (bool, optional): Whether to normalize by magnitude after stft (default=False).
        center (bool, optional): Whether to pad waveform on both sides (default=True).
        pad_mode (str, optional): Controls the padding method used when center is True,
            which can be 'constant', 'edge', 'reflect', 'symmetric' (default='reflect').
        onesided (bool, optional): Controls whether to return half of results to avoid redundancy (default=True).
    """

    def __init__(self, n_fft=400, win_length=None, hop_length=None, pad=0, window="hann", power=2.0,
                 normalized=False, center=True, pad_mode="reflect", onesided=True):
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length if hop_length else self.win_length // 2
        self.pad = pad
        self.window = WindowType(window)
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = BorderType(pad_mode)
        self.onesided = onesided
        self.spectrogram = transforms.Spectrogram(self.n_fft, self.win_length, self.hop_length, self.pad,
                                          self.window, self.power, self.normalized,
                                          self.center, self.pad_mode, self.onesided)

    def __call__(self, waveforms):
        return self.spectrogram(waveforms)


# to-do: test if MelScale works for n_mels=128, MindSpore version >=1.7
class MelSpectrogram(AudioTensorOperation):
    """
    Create a mel-scaled spectrogram from an audio signal.

    Args:
        n_mels (int, optional): Number of mel filterbanks (default=128).
        n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
        sample_rate (int, optional): Sample rate of audio signal (default=16000).
        f_min (float, optional): Minimum frequency (default=0).
        f_max (float, optional): Maximum frequency (default=None, will be set to sample_rate // 2).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use win_length // 2).
        pad (int): Two sided padding of signal (default=0).
        window (str, optional): Window function that is applied/multiplied to each frame/window,
            which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann'). Currently kaiser
            window is not supported on macOS.
        power (float, optional): Exponent for the magnitude spectrogram, which must be greater
            than or equal to 0, e.g., 1 for energy, 2 for power, etc. (default=2.0).
        normalized (bool, optional): Whether to normalize by magnitude after stft (default=False).
        center (bool, optional): Whether to pad waveform on both sides (default=True).
        pad_mode (str, optional): Controls the padding method used when center is True,
            which can be 'constant', 'edge', 'reflect', 'symmetric' (default='reflect').
        onesided (bool, optional): Controls whether to return half of results to avoid redundancy (default=True).
        norm (str, optional): Type of norm, value should be 'slaney' or 'none'. If norm is 'slaney',
            divide the triangular mel weight by the width of the mel band (default='none').
        mel_type (str, optional): Type of scale to use, value should be 'slaney' or 'htk' (default='htk').
    """

    def __init__(self, n_mels=128, n_fft=400, sample_rate=16000, f_min=0.0, f_max=None, win_length=None,
                 hop_length=None, pad=0, window="hann", power=2.0, normalized=False, center=True,
                 pad_mode="reflect", onesided=True, norm="none", mel_type="htk"):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.norm = NormType(norm)
        self.mel_type = MelType(mel_type)
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                       pad=self.pad, window=window, power=self.power, normalized=self.normalized,
                                       center=center, pad_mode=pad_mode, onesided=onesided)
        self.melscale = transforms.MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max,
                                    self.n_fft // 2 + 1, self.norm, self.mel_type)

    def __call__(self, waveforms):
        """
        Args:
            waveforms (np.ndarray): Input audio of dimension (..., time).

        Returns:
            np.ndarray: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """
        specgram = self.spectrogram(waveforms)
        return self.melscale(specgram)


class AmplitudeToDB(AudioTensorOperation):
    """
    Turn a spectrogram from the amplitude/power scale to decibel scale.

    Note:
        The dimension of the input spectrogram to be processed should be (..., freq, time).

    Args:
        stype (str, optional): Scale of the input spectrogram, which can be
            'power' or 'magnitude'. Default: 'power'.
        ref (float, callable, optional): Multiplier reference value for generating
            `db_multiplier`. Default: 1.0. The formula is

            :math:`\text{db_multiplier} = Log10(max(\text{ref}, amin))`.

            `amin` refers to the ower bound to clamp the input waveform, which must
            be greater than zero. Default: 1e-10.
        top_db (float, optional): Minimum cut-off decibels, which must be non-negative. Default: 80.0.

    Raises:
        TypeError: If `stype` is not of type 'power' or 'magnitude'.
        TypeError: If `ref` is not of type float.
        ValueError: If `ref` is not a positive number.
        TypeError: If `top_db` is not of type float.
        ValueError: If `top_db` is not a positive number.
        RuntimeError: If input tensor is not in shape of <..., freq, time>.

    Examples:
        >>> import numpy as np
        >>> waveforms = np.random.random([1, 400 // 2 + 1, 30])
        >>> amplitude_to_db = AmplitudeToDB(stype='power')
        >>> out = amplitude_to_db(waveforms)
    """

    def __init__(self, stype="power", ref=1.0, top_db=80.0):
        self.stype = ScaleType(stype)
        self.ref = ref
        self.amin = 1e-10
        self.top_db = top_db

    def __call__(self, s):
        return amplitude_to_dB(s, self.stype, self.ref, self.amin, self.top_db)


class DBToAmplitude(AudioTensorOperation):
    """
    Turn a dB-scaled spectrogram to the power/amplitude scale.

    Args:
        ref (float, callable): Reference which the output will be scaled by. Can be set to be np.max.
        power (float): If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.

    Examples:
        >>> import numpy as np
        >>> s = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        >>> db_to_amplitude = DBToAmplitude(0.5, 0.5)
        >>> out = db_to_amplitude(s)
    """

    def __init__(self, ref, power):
        self.ref = ref
        self.power = power

    def __call__(self, s):
        return dB_to_amplitude(s, self.ref, self.power)


class MagPhase(AudioTensorOperation):
    """
    Separate a complex-valued spectrogram with shape (..., 2) into its magnitude and phase.

    Args:
        power (float): Power of the norm, which must be non-negative (default=1.0).
    Returns:
        np.ndarray (tuple): A 2-dimension tuple indicating magnitude and phase.

    Examples:
        >>> import numpy as np
        >>> waveforms = np.random.random([2, 4, 2])
        >>> magphase = MagPhase(power=2.0)
        >>> magnitude, phase = magphase(waveforms)
    """

    def __init__(self, power=1.0):
        self.power = power
        self.magphase = transforms.Magphase(self.power)

    def __call__(self, s):
        return self.magphase(s)


class ComputeDeltas(AudioTensorOperation):
    """
    Compute delta coefficients of a spectrogram.

    Args:
        win_length (int): The window length used for computing delta, must be no less than 3 (default=5).
        pad_mode (str): Mode parameter passed to padding, which can be 'constant', 'edge', 'reflect', 'symmetric'
            (default='edge').

            - 'constant', means it fills the border with constant values.

            - 'edge', means it pads with the last value on the edge.

            - 'reflect', means it reflects the values on the edge omitting the last
              value of edge.

            - 'symmetric', means it reflects the values on the edge repeating the last
              value of edge.

    Examples:
        >>> import numpy as np
        >>> waveforms = np.random.random([1, 400//2+1, 30])
        >>> compute_deltas = ComputeDeltas(win_length=7, pad_mode="edge")
        >>> out =compute_deltas(waveforms)
    """

    def __init__(self, win_length=5, pad_mode="edge"):
        self.win_len = win_length
        self.pad_mode = BorderType(pad_mode)
        self.compute_deltas = transforms.ComputeDeltas(self.win_len, self.pad_mode)

    def __call__(self, waveforms):
        return self.compute_deltas(waveforms)


class Trim(AudioTensorOperation):
    """
    Trim the slient segments at the begining and the end of the audio signal.

    Args:
        waveforms (np.ndarray): The audio signal in shape (n,) or (n, n_channel).
        top_db (float): The threshold in decibels below `reference`. The audio segments below this threshold compared to
            `reference` will be considered as silence.
        reference (float, Callable): The reference amplitude. By default, `np.max` is used to serve as the reference
            amplitude.
        frame_length (int): The number of frames per analysis.
        hop_length (int): The number of frames between analysis.
    """

    def __init__(self, top_db=60, reference=np.max, frame_length=2048, hop_length=512):
        super(Trim, self).__init__()
        self.top_db = top_db
        self.reference = reference
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, waveforms):
        """
        Execute the trim operation.

        Args:
            waveforms (np.ndarray): The audio signal in shape (n,) or (n, n_channel).

        Returns:
            np.ndarray, the trimmed signal.
            np.ndarray, the index corresponding to the non-silent region: `wav_trimmed = waveforms[index[0]: index[1]]`
                (for mono) or `wav_trimmed = waveforms[index[0]: index[1], :]`.

        Examples:
            >>> waveforms = np.array([0.01]*1000 + [0.6]*1000 + [-0.6]*1000)
            >>> wav_trimmed, index = trim(waveforms, top_db=10)
            >>> wav_trimmed.shape
            (2488,)
            >>> index[0]
            512
            >>> index[1]
            3000
        """
        return trim(waveforms, self.top_db, self.reference, self.frame_length, self.hop_length)


class Split(AudioTensorOperation):
    """
    Split an audio signal into different non-silent segments.

    Args:
        waveforms (np.ndarray): The audio signal in shape (n,) or (n, n_channel).
        top_db (float): The threshold in decibels below `reference`. The audio segments below this threshold compared to
            `reference` will be considered as silence.
        reference (float, Callable): The reference amplitude. By default, `np.max` is used to serve as the reference
            amplitude.
        frame_length (int): The number of frames per analysis.
        hop_length (int): The number of frames between analysis.
    """

    def __init__(self, top_db=60, reference=np.max, frame_length=2048, hop_length=512):
        super(Split, self).__init__()
        self.top_db = top_db
        self.reference = reference
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, waveforms):
        """
        Execute the trim operation.

        Args:
            waveforms (np.ndarray): The audio signal in shape (n,) or (n, n_channel).

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
        return split(waveforms, self.top_db, self.reference, self.frame_length, self.hop_length)


class ComplexNorm(AudioTensorOperation):
    """
    Compute the norm of complex number sequence.

    Note:
        The dimension of the audio waveform to be processed needs to be (..., complex=2).
        The first dimension represents the real part while the second represents the imaginary.

    Args:
        power (float, optional): Power of the norm, which must be non-negative. Default: 1.0.

    Raises:
        TypeError: If `power` is not of type float.
        ValueError: If `power` is a negative number.
        RuntimeError: If input tensor is not in shape of <..., complex=2>.

    Examples:
        >>> import numpy as np
        >>>
        >>> waveforms = np.arange(1024)
        >>> stft = STFT(return_complex=False)
        >>> inputs_arr = stft(waveforms)
        >>> complex_norm = ComplexNorm(inputs_arr)
    """

    def __init__(self, power=1.0):
        self.complex_norm = transforms.ComplexNorm(power)

    def __call__(self, waveforms):
        """
        Compute the norm of the complex input.

        Args:
            waveforms (np.ndarray): Array shape of (..., 2).

        Returns:
            np.ndarray, norm of the input signal.
        """
        return self.complex_norm(waveforms)


class Angle(AudioTensorOperation):
    """
    Calculate the angle of the complex number sequence of shape (..., 2).
    The first dimension represents the real part while the second represents the imaginary.
    """
    def __init__(self):
        self.angle = transforms.Angle()

    def __call__(self, x):
        return self.angle(x)


class FrequencyMasking(AudioTensorOperation):
    """
    Apply masking to a spectrogram in the frequency domain.

    Args:
        iid_masks (bool, optional): Whether to apply different masks to each example (default=false).
        frequency_mask_param (int, optional): Maximum possible length of the mask, range: [0, freq_length] (default=0).
            Indices uniformly sampled from [0, frequency_mask_param].
        mask_start (int, optional): Mask start takes effect when iid_masks=true,
            range: [0, freq_length-frequency_mask_param] (default=0).
        mask_value (float, optional): Mask value (default=0.0).
    """
    def __init__(self, iid_masks=False, frequency_mask_param=0, mask_start=0, mask_value=0.0):
        self.iid_masks = iid_masks
        self.frequency_mask_param = frequency_mask_param
        self.mask_start = mask_start
        self.mask_value = mask_value
        self.frequency_masking = transforms.FrequencyMasking(self.iid_masks, self.frequency_mask_param,
                                                     self.mask_start, self.mask_value)

    def __call__(self, x):
        return self.frequency_masking(x)


class TimeMasking(AudioTensorOperation):
    """
    Apply masking to a spectrogram in the time domain.

    Args:
        iid_masks (bool, optional): Whether to apply different masks to each example (default=false).
        time_mask_param (int, optional): Maximum possible length of the mask, range: [0, time_length] (default=0).
            Indices uniformly sampled from [0, time_mask_param].
        mask_start (int, optional): Mask start takes effect when iid_masks=true,
            range: [0, time_length-time_mask_param] (default=0).
        mask_value (float, optional): Mask value (default=0.0).
    """
    def __init__(self, iid_masks=False, time_mask_param=0, mask_start=0, mask_value=0.0):
        self.iid_masks = iid_masks
        self.time_mask_param = time_mask_param
        self.mask_start = mask_start
        self.mask_value = mask_value
        self.time_masking = transforms.TimeMasking(self.iid_masks, self.time_mask_param, self.mask_start, self.mask_value)

    def __call__(self, x):
        return self.time_masking(x)


class SlidingWindowCmn(AudioTensorOperation):
    """
    Apply sliding-window cepstral mean (and optionally variance) normalization per utterance.

    Args:
        cmn_window (int, optional): Window in frames for running average CMN computation (default=600).
        min_cmn_window (int, optional): Minimum CMN window used at start of decoding (adds latency only at start).
            Only applicable if center is False, ignored if center is True (default=100).
        center (bool, optional): If True, use a window centered on the current frame. If False, window is
            to the left. (default=False).
        norm_vars (bool, optional): If True, normalize variance to one. (default=False).
    """
    def __init__(self, cmn_window=600, min_cmn_window=100, center=False, norm_vars=False):
        self.cmn_window = cmn_window
        self.min_cmn_window = min_cmn_window
        self.center = center
        self.norm_vars = norm_vars
        self.sliding_window_cmn = transforms.SlidingWindowCmn(self.cmn_window, self.min_cmn_window, self.center, self.norm_vars)

    def __call__(self, x):
        return self.sliding_window_cmn(x)


class MelScale(AudioTensorOperation):
    def __init__(self, n_mels=128, sample_rate=16000, f_min=0, f_max=None, n_stft=201, norm=NormType.NONE,
                 mel_type=MelType.HTK):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.n_stft = n_stft
        self.norm = norm
        self.mel_type = mel_type
        self.mel_scale = transforms.MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_stft,
                                     self.norm, self.mel_type)

    def __call__(self, x):
        return self.mel_scale(x)


class PreEmphasis(AudioTensorOperation):
    def __init__(self, alpha=0.97):
        self.alpha = alpha

    def __call__(self, x):
        return np.append(x[0], x[1:] - self.alpha * x[:-1])
