import numpy as np
import random
import mindspore as ms
import mindspore.dataset.audio as msaudio
from mindspore.nn import Conv1d
from .io import read
from .spectrum import compute_amplitude
from .processing import rescale

__all__ = [
    'frequencymasking',
    'timemasking',
    'reverberate',
    'add_noise',
    'add_reverb',
]


def frequencymasking(waveform, iid_masks=False, frequency_mask_param=0, mask_start=0, mask_value=0.0):
    """
    Apply masking to a spectrogram in the frequency domain.

    Args:
        waveforms (np.ndarray): A waveform to mask, shape should be
                                    `[time]` or `[batch, time]` or `[batch, time, channels]`
        iid_masks (bool): Whether to apply different masks to each example (default=false).
        frequency_mask_param (int): Maximum possible length of the mask, range: [0, freq_length] (default=0).
            Indices uniformly sampled from [0, frequency_mask_param].
        mask_start (int): Mask start takes effect when iid_masks=true,
            range: [0, freq_length-frequency_mask_param] (default=0).
        mask_value (float): Mask value (default=0.0).
    Returns:
        spectrogram(np.ndarray): A spectrogram applyed frequencymasking
    Examples:
        >>> import mindaudio.data.spectrum as spectrum
        >>> import mindaudio.data.augment as augment
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> orignal = spectrum.spectrogram(waveform)
        >>> masked = augment.frequencymasking(orignal, frequency_mask_param=80)
    """
    frequency_masking = msaudio.FrequencyMasking(iid_masks, frequency_mask_param, mask_start, mask_value)

    return frequency_masking(waveform)


def timemasking(waveform, iid_masks=False, frequency_mask_param=0, mask_start=0, mask_value=0.0):
    """
    Apply masking to a spectrogram in the time domain.

    Args:
        waveforms (np.ndarray): A waveform to mask, shape should be
                                    `[time]` or `[batch, time]` or `[batch, time, channels]`
        iid_masks (bool): Whether to apply different masks to each example (default=false).
        frequency_mask_param (int): Maximum possible length of the mask, range: [0, freq_length] (default=0).
            Indices uniformly sampled from [0, frequency_mask_param].
        mask_start (int): Mask start takes effect when iid_masks=true,
            range: [0, freq_length-frequency_mask_param] (default=0).
        mask_value (float): Mask value (default=0.0).
    Examples:
        >>> import mindaudio.data.spectrum as spectrum
        >>> import mindaudio.data.augment as augment
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> orignal = spectrum.spectrogram(waveform)
        >>> masked = augment.timemasking(orignal, frequency_mask_param=80)

    """
    time_masking = msaudio.TimeMasking(iid_masks, frequency_mask_param, mask_start, mask_value)

    return time_masking(waveform)


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

    Examples:
        >>> import mindaudio.data.augment as augment
        >>> read_wav_dir = './samples/ASR/BAC009S0002W0122.wav'
        >>> samples, _ = io.read(read_wav_dir)
        >>> rirs, _ = io.read('./samples/ASR/1089-134686-0001.wav')
        >>> addnoise = augment.reverberate(samples, rirs)
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


def rms_normalize(samples):
    """
    Power-normalise samples.

    Args:
        samples (np.ndarray): the shape should be (..., time)

    Returns:
        samples(np.ndarray):samples normalised.
        """
    rms = np.sqrt(np.square(samples).mean(keepdims=True))
    return samples / (rms + 1e-8)


def caculate_rms(samples):
    """
    Caculate rms.

    Args:
        samples (np.ndarray): the shape should be (..., time)

    Returns:
        value(float):rms value
    """
    rms = np.sqrt(np.square(samples).mean(axis=-1, keepdims=False))
    return rms


def add_noise(samples, backgroundlist, min_snr_in_db, max_snr_in_db, mix_prob=1.0):
    """
    add background noise.

    Args:
        samples (np.ndarray): The audio signal to perform convolution on.The shape should be
            `[time]` or `[batch, time]` or `[batch, channels, time]`
        backgroundlist (list): List of paths to background audio files.
        min_snr_in_db(int): nimimum SNR in dB
        max_snr_in_db(int): maximum SNR in dB
        mix_prob(float): The probablity that the audio signals will be mix.

    Returns:
        samples(np.ndarray):samples added background noise

    Examples:
        >>> import numpy as np
        >>> samples = np.random.rand(10, 1, 200960) - 0.5
        >>> background_list = ['./samples/ASR/1089-134686-0000.wav']
        >>> noise_added_saples = add_noise(samples, background_list, 3, 30)
    """
    if np.random.rand(1) > mix_prob:
        return samples

    dimension_of_samples = len(samples.shape)
    if dimension_of_samples > 3:
        raise NotImplementedError

    # if inputs are mono tensors we reshape to 1, samples
    if dimension_of_samples == 1:
        samples = np.expand_dims(np.expand_dims(samples, 0), 1)
    elif dimension_of_samples == 2:
        samples = np.expand_dims(samples, 1)
    batch, chanel, sample_lenth = samples.shape

    missing_num_samples = sample_lenth
    while missing_num_samples > 0:
        background_path = random.choice(backgroundlist)
        noise_audio, sr = read(background_path)
        background_num_samples = len(noise_audio)

        if background_num_samples > missing_num_samples:
            num_samples = missing_num_samples
            background_samples = rms_normalize(noise_audio[:num_samples])
            missing_num_samples = 0
        else:
            background_samples = rms_normalize(noise_audio)
            missing_num_samples -= background_num_samples

        if 'pieces' in vars():
            pieces = np.append(pieces, background_samples)
        else:
            pieces = background_samples

    background = rms_normalize(pieces.reshape(1, sample_lenth))

    sample_rms = caculate_rms(samples)
    snr = np.random.uniform(min_snr_in_db, max_snr_in_db, 1)
    background_scale = sample_rms/(10 ** (snr/20))
    background_noise = (np.expand_dims(background, axis=0)) * (np.expand_dims(background_scale, axis=2))
    samples_added_noise = samples + background_noise

    if dimension_of_samples == 1:
        samples_added_noise = samples_added_noise.squeeze(axis=1).squeeze(axis=0)
    elif dimension_of_samples == 2:
        samples_added_noise = samples_added_noise.squeeze(axis=1)

    return samples_added_noise


def add_reverb(samples, rirlist, reverb_prob=1.0):
    """
        add reverb.

        Args:
            samples (np.ndarray): The audio signal to perform convolution on. The shape should be
            `[time]` or `[batch, time]` or `[batch, channels, time]`
            rirlist (list): List of paths to RIR files.
            reverb_prob(float): The chance that the audio signal will be reverbed.

        Returns:
            samples(np.ndarray):samples added reverb

        Examples:
            >>> import numpy as np
            >>> samples = np.random.rand(10, 1, 200960) - 0.5
            >>> background_list = ['./samples/rir/air_binaural_aula_carolina_0_1_3_0_3_16k.wav']
            >>> addrir = augment.add_reverb(samples, rir_list, 1.0)

    """
    if np.random.rand(1) > reverb_prob:
        return samples

    orig_shapelen = len(samples.shape)

    if orig_shapelen > 3:
        raise NotImplementedError
    elif orig_shapelen == 2:
        samples = np.expand_dims(samples, axis=2)
    elif orig_shapelen == 3:
        batch, chanel, times = samples.shape
        samples = np.expand_dims(samples.reshape(batch*chanel, times), axis=2)

    rir_path = random.choice(rirlist)
    rir_waveform, sr = read(rir_path)
    res = reverberate(samples, rir_waveform)

    if orig_shapelen == 3:
        res = np.squeeze(res, axis=2).reshape(batch, chanel, times)
    elif orig_shapelen == 2:
        res = np.squeeze(res, axis=2)
    elif orig_shapelen == 1:
        res = np.squeeze(res, axis=0)

    return res









