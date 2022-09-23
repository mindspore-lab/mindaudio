import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.dataset.audio.transforms import AudioTensorOperation
from mindspore.dataset.audio.utils import WindowType, NormMode, create_dct
import mindspore.dataset.audio.transforms as tf
from mindaudio.data.features.processing import AmplitudeToDB, ComputeDeltas, MelSpectrogram


__all__ = [
    'SpectralCentroid',
    'ContextWindow',
    'Fbank',
    'MFCC',
]


class SpectralCentroid(AudioTensorOperation):
    """
    Create a spectral centroid from an audio signal.

    Args:
        sample_rate (int): Sampling rate of the waveform, e.g. 44100 (Hz).
        n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins (default=400).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use win_length // 2).
        pad (int, optional): Two sided padding of signal (default=0).
        window (str, optional): Window function that is applied/multiplied to each frame/window,
        which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann'). Currently kaiser
            window is not supported on macOS.
    """

    def __init__(self, sample_rate, n_fft=400, win_length=None, hop_length=None, pad=0, window="hann"):
        self.sample_rate = sample_rate
        self.pad = pad
        self.window = WindowType(window)
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length if hop_length else self.win_length // 2
        self.spectral_centroid = tf.SpectralCentroid(self.sample_rate, self.n_fft, self.win_length, self.hop_length,
                                                     self.pad, self.window)

    def __call__(self, waveforms):
        return self.spectral_centroid(waveforms)


class ContextWindow(AudioTensorOperation):
    """
    Create a context window from an audio signal to gather multiple time step in a single feature vector.

    Args:
        left_frames (int): Number of past frames to collect.
        left_frames (int): Number of future frames to collect.
    """

    def __init__(self, left_frames=0, right_frames=0):
        super(ContextWindow, self).__init__()
        self.left_frames = left_frames
        self.right_frames = right_frames
        self.context_size = self.left_frames + self.right_frames + 1
        self.kernel_size = 2 * max(self.left_frames, self.right_frames) + 1
        self.shift = right_frames - left_frames
        self.max_frame = max(self.left_frames, self.right_frames)
        # Kernel definition
        self.kernel = np.eye(self.context_size, self.kernel_size, dtype=np.float32)
        if self.shift > 0:
            self.kernel = np.roll(self.kernel, self.shift, 1)
        self.first_call = True

    def __call__(self, waveforms):
        """Returns the array with the surrounding context.

        Args:
            waveforms(np.ndarray): Single-channel or multi-channel time-series audio signals with shape [freq, time],
                                   [batch, freq, time] or [batch, channel, freq, time].
        Returns:
            np.array: Aggregated feature vector by gathering the past and future time steps. The feature with shape
                      [freq, time], [batch, freq, time] or [batch, channel, freq, time].
        """

        input_shape = waveforms.shape
        # Considering multi-channel case
        if len(input_shape) == 2:
            x = np.expand_dims(waveforms, 0)
        elif len(input_shape) == 3:
            x = waveforms
        elif len(input_shape) == 4:
            x = waveforms.transpose((0, 2, 3, 1))
        else:
            raise TypeError("Input dimension must be 2, 3 or 4, but got {}".format(len(input_shape)))

        if self.first_call:
            self.first_call = False
            tile = np.tile(self.kernel, (x.shape[1], 1, 1))
            tile = tile.reshape((x.shape[1] * self.context_size, self.kernel_size, ))
            self.kernel = np.expand_dims(tile, 1)

        x_shape = x.shape
        if len(x_shape) == 4:
            x = x.reshape((x_shape[0] * x_shape[2], x_shape[1], x_shape[3]))

        # Computing context using the estimated kernel
        in_channel, out_channel = x_shape[1], self.kernel.shape[0]
        self.conv = nn.Conv1d(in_channel, out_channel, self.kernel_size,
                              padding=self.max_frame, pad_mode='pad',
                              group=x.shape[1], weight_init=Tensor(self.kernel))
        x_tensor = Tensor(x, ms.float32)
        context = self.conv(x_tensor)
        # Retrieving the original dimensionality for multi-channel case
        if len(x_shape) == 4:
            context = context.reshape((x_shape[0], context.shape[1],
                                       x_shape[2], context.shape[-1])
                                      )
            context = context.transpose((0, 3, 1, 2))

        if len(x_shape) == 2:
            context = np.squeeze(context, 0)

        return context.asnumpy()


#to-do: check the difference between melscale in MindSpore and fbank matrix in Speechbrain
class Fbank(AudioTensorOperation):
    """Generate filter bank features.

    Args:
        deltas (bool): Whether or not to append derivatives and second derivatives to the features (default: False).
        context (bool): Whether or not to append forward and backward contexts to the features (default: False).
        n_mels (int): Number of Mel filters (default: 40).
        n_fft (int): Number of samples used in each stft (default: 400).
        sample_rate (int): Sampling rate for the input waveforms (default: 160000).
        f_min (float, optional): Minimum frequency (default=0).
        f_max (float, optional): Maximum frequency (default=None, will be set to sample_rate // 2).
        left_frames (int): Number of frames  left context to collect (default: 5).
        right_frames (int): Number of past frames to collect (default: 5).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use win_length // 2).
        window (str, optional): Window function that is applied/multiplied to each frame/window,
            which can be 'bartlett', 'blackman', 'hamming', 'hann' or 'kaiser' (default='hann'). Currently kaiser
                window is not supported on macOS.

    Example:
        >>> import numpy as np
        >>> inputs = np.random.random([10, 16000])
        >>> fbanks = Fbank()
        >>> feats = fbanks(inputs)
        >>> feats.shape
        (10, 40, 101)
    """

    def __init__(self, deltas=False, context=False, n_mels=40, n_fft=400, sample_rate=16000,
                 f_min=0.0, f_max=None, left_frames=5, right_frames=5, win_length=None,
                 hop_length=None, window="hann"):
        self.deltas = deltas
        self.context = context
        self.melspec = MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, f_min=f_min,
                                      f_max=f_max, win_length=win_length, hop_length=hop_length, window=window)
        self.amplitude_to_dB = AmplitudeToDB(stype="power", ref=1.0, top_db=80.0)
        self.compute_deltas = ComputeDeltas()
        self.context_window = ContextWindow(left_frames=left_frames, right_frames=right_frames)

    def __call__(self, waveforms):
        """Returns a set of FBank features generated from the input waveforms.

        Args:
            waveforms(np.ndarray): A batch of audio signals with shape [time], [batch, time] or [batch, channel, time].
        Returns:
            np.ndarray: Mel-frequency cepstrum coefficients with shape [freq, time], [batch, freq, time] or
                        [batch, channel, freq, time].
        """
        melspcgram = self.melspec(waveforms)
        fbanks = self.amplitude_to_dB(melspcgram)
        if self.deltas:
            delta1 = self.compute_deltas(fbanks)
            delta2 = self.compute_deltas(delta1)
            fbanks = np.concatenate((fbanks, delta1, delta2), axis=-2)
        if self.context:
            fbanks = self.context_window(fbanks)
        return fbanks


#to-do: check the difference between melscale in MindSpore and fbank matrix in Speechbrain
class MFCC(AudioTensorOperation):
    """Generate Mel-frequency cepstrum coefficients (MFCC) features from input audio signal.

    Args:
        deltas (bool): Whether or not to append derivatives and second derivatives to the features (default: False).
        context (bool): Whether or not to append forward and backward contexts to the features (default: False).
        n_mels (int): Number of Mel filters (default: 23).
        n_mfcc (int): Number of Mel-frequency cepstrum coefficients (default: 20).
        n_fft (int): Number of samples used in each stft (default: 400).
        sample_rate (int): Sampling rate for the input waveforms (default: 160000).
        f_min (float, optional): Minimum frequency (default=0).
        f_max (float, optional): Maximum frequency (default=None, will be set to sample_rate // 2).
        left_frames (int): Number of frames  left context to collect (default: 5).
        right_frames (int): Number of past frames to collect (default: 5).
        win_length (int, optional): Window size (default=None, will use n_fft).
        hop_length (int, optional): Length of hop between STFT windows (default=None, will use win_length // 2).
        norm (str, optional): Normalization mode, can be "none" or "orhto" (default="none").
        log_mels (bool, optional): Whether to use log-mel spectrograms instead of db-scaled (default=False).

    Example:
        >>> import numpy as np
        >>> inputs = np.random.random([10, 16000])
        >>> mfccs = MFCC()
        >>> feats = mfccs(inputs)
        >>> feats.shape
        (10, 660, 101)
    """

    def __init__(self, deltas=True, context=True, n_mels=23, n_mfcc=20, n_fft=400,
                 sample_rate=16000, f_min=0.0, f_max=None, left_frames=5, right_frames=5,
                 win_length=None, hop_length=None, norm="ortho", log_mels=False):
        self.deltas =deltas
        self.context = context
        self.n_mfcc = n_mfcc
        self.norm = NormMode(norm)
        self.log_mels = log_mels
        self.melspecgram = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels,
                                          f_min=f_min, f_max=f_max, win_length=win_length,
                                          hop_length=hop_length)

        if self.n_mfcc > self.melspecgram.n_mels:
            raise ValueError('The number of MFCC coefficients must be no more than # mel bins.')
        self.dct = create_dct(n_mfcc=self.n_mfcc, n_mels=n_mels, norm=self.norm)
        self.compute_deltas = ComputeDeltas()
        self.context_window = ContextWindow(left_frames=left_frames, right_frames=right_frames)
        self.amplitude_to_dB = AmplitudeToDB(stype="power", ref=1.0, top_db=80.0)

    def __call__(self, waveforms):
        """Returns a set of MFCC features generated from the input waveforms.

        Args:
            waveforms(np.ndarray): A batch of audio signals with shape [time], [batch, time] or [batch, channel, time].
        Returns:
            np.ndarray: Mel-frequency cepstrum coefficients with shape [freq, time], [batch, freq, time]
                        or [batch, channel, freq, time].
        """
        melspecgram = self.melspecgram(waveforms)
        if self.log_mels:
            melspecgram = np.log(melspecgram + 1e-6)
        else:
            melspecgram = self.amplitude_to_dB(melspecgram)
        melspecgram_shape = melspecgram.shape
        # Considering multi-channel case
        # (..., time, n_mels) dot (n_mels, n_mfcc) -> (..., n_mfcc, time)
        if len(melspecgram_shape) == 2:
            mfccs = np.matmul(melspecgram.transpose((1, 0)), self.dct).transpose((1, 0))
        elif len(melspecgram_shape) == 3:
            mfccs = np.matmul(melspecgram.transpose((0, 2, 1)), self.dct).transpose((0, 2, 1))
        elif len(melspecgram_shape) == 4:
            mfccs = np.matmul(melspecgram.transpose((0, 1, 3, 2)), self.dct).transpose((0, 1, 3, 2))
        else:
            raise TypeError("Unsupported MelSpectrogram shape {}".format(len(melspecgram_shape)))

        if self.deltas:
            delta1 = self.compute_deltas(mfccs)
            delta2 = self.compute_deltas(delta1)
            mfccs = np.concatenate((mfccs, delta1, delta2), axis=-2)
        if self.context:
            mfccs = self.context_window(mfccs)
        return mfccs
