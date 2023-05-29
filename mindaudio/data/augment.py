import random

import mindspore as ms
import mindspore.dataset.audio as msaudio
import numpy as np
from mindspore.nn import Conv1d

from .filters import notch_filter
from .io import read
from .processing import resample, rescale
from .spectrum import _pad_shape, compute_amplitude, dB_to_amplitude, istft, stft

__all__ = [
    "frequencymasking",
    "timemasking",
    "reverberate",
    "add_noise",
    "add_reverb",
    "add_babble",
    "drop_freq",
    "speed_perturb",
    "drop_chunk",
    "time_stretch",
    "pitch_shift",
]


def frequencymasking(
    waveform, iid_masks=False, frequency_mask_param=0, mask_start=0, mask_value=0.0,
):
    """
    Apply masking to a spectrogram in the frequency domain.

    Args:
        waveforms (np.ndarray): A waveform to mask, shape should be
            `[time]` or `[batch, time]` or `[batch, time, channels]`
        iid_masks (bool): Whether to apply different masks to each example
            (default=false).
        frequency_mask_param (int): Maximum possible length of the mask,
            range: [0, freq_length] (default=0).
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
    frequency_masking = msaudio.FrequencyMasking(
        iid_masks, frequency_mask_param, mask_start, mask_value
    )

    return frequency_masking(waveform)


def timemasking(
    waveform, iid_masks=False, frequency_mask_param=0, mask_start=0, mask_value=0.0,
):
    """
    Apply masking to a spectrogram in the time domain.

    Args:
        waveforms (np.ndarray): A waveform to mask, shape should be
            `[time]` or `[batch, time]` or `[batch, time, channels]`
        iid_masks (bool): Whether to apply different masks to each example
            (default=false).
        frequency_mask_param (int): Maximum possible length of the mask,
            range: [0, freq_length] (default=0).
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
    time_masking = msaudio.TimeMasking(
        iid_masks, frequency_mask_param, mask_start, mask_value
    )

    return time_masking(waveform)


def reverberate(waveforms, rir_waveform, rescale_amp="avg"):
    """
    Reverberate a given signal with given a Room Impulse Response (RIR).
    It performs convolution between RIR and signal,
    but without changing the original amplitude of the signal.

    Args:
        waveforms (np.ndarray): The audio signal to reverberate.
        rir_waveform (np.ndarray): The Room Impulse Response signal.
        rescale_amp (str): Whether reverberated signal is rescaled (None)
            and with respect either to original signal
            "peak" amplitude or "avg" average amplitude.
            Options: [None, "avg", "peak"].

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
    orig_amplitude = compute_amplitude(waveforms, waveforms.shape[1], rescale_amp)

    # Compute index of the direct signal, so we can preserve alignment
    # value_max = np.max(np.abs(rir_waveform), axis=1, keepdims=True)
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


def convolve1d(
    waveforms,
    kernel,
    padding=0,
    pad_type="constant",
    stride=1,
    groups=1,
    use_fft=True,
    rotation_index=0,
):
    """Use mindspore.conv1d to perform 1d padding and convolution.

    Args:
        waveforms (np.ndarray): The audio signal to perform convolution on.
        kernel (np.ndarray): The filter kernel to apply during convolution.
        padding (int, tuple): The padding size to apply at
            left side and right side.
        pad_type (str): The type of padding to use.
            Options: ["constant", "edge"].
        stride (int): The number of units to stride for the
            convolution operations. If `use_fft` is True, this will not have
            effects.
        groups (int): This option is passed to `conv1d` to split the input
            into groups for convolution. Input channels
            should be divisible by the number of groups.
        use_fft (bool): When `use_fft` is passed `True`, then compute the
            convolution in the spectral domain using complex multiply.
            This is more efficient on CPU when the size of the kernel is large.
            WARNING: Without padding, circular convolution occurs.
            This makes little difference in the case of reverberation,
            but may make more difference with different kernels.
        rotation_index (int): This option only applies if `use_fft` is true.
            If so, the kernel is rolled by this amount
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
        conv1d = Conv1d(
            1,
            1,
            kernel_size=kernel.shape[-1],
            stride=stride,
            group=groups,
            padding=0,
            pad_mode="valid",
        )
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
        samples (np.ndarray): The audio signal to perform convolution on.The
        shape should be`[time]` or `[batch, time]` or `[batch, channels, time]`
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
    pieces = None
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

        if pieces is not None:
            pieces = np.append(pieces, background_samples)
        else:
            pieces = background_samples

    background = rms_normalize(pieces.reshape(1, sample_lenth))

    sample_rms = caculate_rms(samples)
    snr = np.random.uniform(min_snr_in_db, max_snr_in_db, 1)
    background_scale = sample_rms / (10 ** (snr / 20))
    background_noise = (np.expand_dims(background, axis=0)) * (
        np.expand_dims(background_scale, axis=2)
    )
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
        samples (np.ndarray): The audio signal to perform convolution on.
        The shape should be `[time]` or `[batch, time]`
        or `[batch, channels, time]`
        rirlist (list): List of paths to RIR files.
        reverb_prob(float): The chance that the audio signal will be
            reverbed.

    Returns:
        samples(np.ndarray):samples added reverb

    Examples:
        >>> import numpy as np
        >>> samples = np.random.rand(10, 1, 200960) - 0.5
        >>> background_list =
        ['./samples/rir/air_binaural_aula_carolina_0_1_3_0_3_16k.wav']
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
        samples = np.expand_dims(samples.reshape(batch * chanel, times), axis=2)

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


def add_babble(
    waveforms, lengths, speaker_count=3, snr_low=0, snr_high=0, mix_prob=1.0
):
    """
        Simulate babble noise by mixing the signals in a batch.

        Args:
            waveforms(np.ndarray): A batch of audio signals to process,
                with shape `[batch, time]` or `[batch, time, channels]`.
            lengths(np.ndarray): The length of each audio in the batch,
                with shape `[batch]`.
            speaker_count(int): The number of signals to mix with the
                original signal.
            snr_low(int): The low end of the mixing ratios, in decibels.
            snr_high(int): The high end of the mixing ratios, in decibels.
            mix_prob(float): The probability that the batch of signals will
                bemixed with babble noise.By default, every signal is mixed.

        Returns:
            waveforms(np.ndarray):array with processed waveforms.

        Examples:
            >>> import mindaudio.data.io as io
            >>> import mindaudio.data.augment as augment
            >>> wav_list = ['./samples/ASR/BAC009S0002W0122.wav',
            >>>        './samples/ASR/BAC009S0002W0123.wav',
            >>>        './samples/ASR/BAC009S0002W0124.wav',]
            >>> wav_num = 0
            >>> maxlen = 0
            >>> lenlist = []
            >>> for wavdir in wav_list:
            >>>     wav, _ = io.read(wavdir)
            >>>     wavlen = len(wav)
            >>>     lenlist.append(wavlen)
            >>>     maxlen = max(wavlen, maxlen)
            >>>     if wav_num == 0:
            >>>         waveforms = np.expand_dims(wav, axis=0)
            >>>     else:
            >>>         wav = np.expand_dims(np.pad(wav, (0, maxlen-wavlen),\
            'constant'), axis=0)
            >>>         waveforms = np.concatenate((waveforms, wav), axis=0)
            >>>     wav_num += 1
            >>> lengths = np.array(lenlist)/maxlen
            >>> noisy_mindaudio = augment.add_babble(waveforms, lengths, \
            speaker_count=3, snr_low=0, snr_high=0)

    """
    babbled_waveform = waveforms.copy()
    lengths = np.expand_dims(lengths * waveforms.shape[1], axis=1)
    batch_size = len(waveforms)

    if np.random.rand(1) > mix_prob:
        return babbled_waveform

    # Pick an SNR and use it to compute the mixture amplitude factors
    clean_amplitude = compute_amplitude(waveforms, lengths)
    SNR = np.random.rand(batch_size, 1)
    SNR = SNR * (snr_high - snr_low) + snr_low
    noise_amplitude_factor = 1 / (dB_to_amplitude(SNR, 1, 1) + 1)
    new_noise_amplitude = noise_amplitude_factor * clean_amplitude

    # Scale clean signal appropriately
    babbled_waveform *= 1 - noise_amplitude_factor

    # For each speaker in the mixture, roll and add
    babble_waveform = np.roll(waveforms, 1, axis=0)
    babble_len = np.roll(lengths, 1, axis=0)
    for i in range(1, speaker_count):
        babble_waveform += np.roll(waveforms, 1 + i, axis=0)
        babble_len = np.maximum(babble_len, np.roll(babble_len, 1, axis=0))

    # Rescale and add to mixture
    babble_amplitude = compute_amplitude(babble_waveform, babble_len)
    babble_waveform *= new_noise_amplitude / (babble_amplitude + 1e-14)
    babbled_waveform += babble_waveform

    return babbled_waveform


def drop_freq(
    waveforms,
    drop_freq_low=1e-14,
    drop_freq_high=1,
    drop_count_low=1,
    drop_count_high=2,
    drop_width=0.05,
    drop_prob=1,
):
    """
    Drops a random frequency from the signal.To teach models to learn to rely
    on all parts of the signal,not just a few frequency bands.

    Args:
        waveforms(np.ndarray): A batch of audio signals to process,
        with shape `[batch, time]` or`[batch, time, channels]`.
        drop_freq_low(float): The low end of frequencies that can be dropped,
            as a fraction of the sampling rate / 2.
        drop_freq_high(float): The high end of frequencies that can be dropped,
            as a fraction of the sampling rate / 2.
        drop_count_low(int): The low end of number of frequencies that could
            be dropped.
        drop_count_high(int): The high end of number of frequencies that could
            be dropped.
        drop_width(float): The width of the frequency band to drop, as a
            fraction of the sampling_rate / 2.
        drop_prob(float): The probability that the batch of signals will have a
            frequency dropped. By default,every batch has frequencies dropped.

    Returns:
        ndarray of shape `[batch, time]` or `[batch, time, channels]`

    Examples:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.augment as augment
        >>> signal = io.read('./samples/ASR/1089-134686-0000.wav')
        >>> dropped_signal_mindaudio = augment.drop_freq(signal)

    """
    # Don't drop (return early) 1-`drop_prob` portion of the batches
    orig_shapelen = len(waveforms.shape)
    dropped_waveform = waveforms.copy()
    if np.random.rand(1) > drop_prob:
        return dropped_waveform

    # Add channels dimension
    if len(waveforms.shape) == 1:
        dropped_waveform = np.expand_dims(np.expand_dims(dropped_waveform, 0), 2)
    elif len(waveforms.shape) == 2:
        dropped_waveform = np.expand_dims(dropped_waveform, axis=2)

    # Pick number of frequencies to drop
    drop_count = np.random.randint(
        low=drop_count_low, high=drop_count_high + 1, size=(1,)
    )
    drop_count = drop_count[0]

    # Pick a frequency to drop
    drop_range = drop_freq_high - drop_freq_low
    drop_frequency = np.random.rand(drop_count) * drop_range + drop_freq_low

    # Filter parameters
    filter_length = 101
    pad = filter_length // 2

    # Start with delta function
    drop_filter = np.zeros([1, filter_length, 1])
    drop_filter[0, pad, 0] = 1

    # Subtract each frequency
    for frequency in drop_frequency:
        notch_kernel = notch_filter(frequency, filter_length, drop_width,)
        drop_filter = convolve1d(drop_filter, notch_kernel, pad)

    # Apply filter
    dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

    # Remove channels dimension if added
    if orig_shapelen == 2:
        dropped_waveform = np.squeeze(dropped_waveform, axis=2)
    elif orig_shapelen == 1:
        dropped_waveform = np.squeeze(np.squeeze(dropped_waveform, axis=2), axis=0)
    return dropped_waveform


def speed_perturb(waveform, orig_freq, speeds=[90, 100, 110], perturb_prob=1.0):
    """
    Slightly speed up or slow down an audio signal.Resample the audio signal
        at a rate that is similar to the original rate, to achieve a slightly
        slower or slightly faster signal.

    Args:
        waveform(np.ndarray): Shape should be `[batch, time]` or
            `[batch, time, channels]`.
        orig_freq(int): The frequency of the original signal.
        speeds(list): The speeds that the signal should be changed to, as a
            percentage of the original signal (i.e. `speeds` is
            divided by 100 to get a ratio).
        perturb_prob(float): The chance that the batch will be speed-perturbed.
            By default, every batch is perturbed.

    Returns:
        perturbed_waveform(np.ndarray): Shape `[batch, time]` or
        `[batch, time, channels]`.

    Example:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.augment as augment
        >>> signal = io.read('./samples/ASR/1089-134686-0000.wav')
        >>> perturbed_mindaudio = augment.speed_perturb(signal, \
        orig_freq=16000, speeds=[90])
    """

    # Don't perturb (return early) 1-`perturb_prob` portion of the batches
    if np.random.rand(1) > perturb_prob:
        return waveform.copy()

    # Perform a random perturbation
    samp_index = np.random.randint(0, len(speeds), (1,))[0]
    speed = speeds[samp_index]
    new_freq = orig_freq * speed // 100
    perturbed_waveform = resample(waveform, orig_freq, new_freq)
    return perturbed_waveform


def drop_chunk(
    waveforms,
    lengths,
    drop_length_low=100,
    drop_length_high=1000,
    drop_count_low=1,
    drop_count_high=10,
    drop_start=0,
    drop_end=None,
    drop_prob=1,
    noise_factor=0.0,
):
    """
    This class drops portions of the input signal.Using `drop_chunk` as an
    augmentation strategy helps a models learn to rely on all parts of the
    signal, since it can't expect a given part to be present.

    Args:
        waveforms(np.ndarray): Shape should be `[batch, time]` or
            `[batch, time, channels]`.
        lengths(np.ndarray): Shape should be a single dimension, `[batch]`.
        drop_length_low(int): The low end of lengths for which to set the
            signal to zero, in samples.
        drop_length_high(int): The high end of lengths for which to set the
            signal to zero, in samples.
        drop_count_low(int): The low end of number of times that the signal
            can be dropped to zero.
        drop_count_high(int): The high end of number of times that the signal
            can be dropped to zero.
        drop_start(int): The first index for which dropping will be allowed.
        drop_end(int): The last index for which dropping will be allowed.
        drop_prob(float): The probability that the batch of signals will have
            a portion dropped. By default, every
        batch has portions dropped.
        noise_factor(float): The factor relative to average amplitude of an
            utterance to use for scaling the white
        noise inserted. 1 keeps the average amplitude the same, while 0
            inserts all 0's.

    Returns:
        dropped_waveform(np.ndarray): Shape `[batch, time]` or
        `[batch, time, channels]`

    Example:
        >>> import mindaudio.data.io as io
        >>> import mindaudio.data.augment as augment
        >>> wav_list = ['./samples/ASR/BAC009S0002W0122.wav',
        >>>        './samples/ASR/BAC009S0002W0123.wav',
        >>>        './samples/ASR/BAC009S0002W0124.wav',]
        >>> wav_num = 0
        >>> maxlen = 0
        >>> lenlist = []
        >>> for wavdir in wav_list:
        >>>     wav, _ = io.read(wavdir)
        >>>     wavlen = len(wav)
        >>>     lenlist.append(wavlen)
        >>>     maxlen = max(wavlen, maxlen)
        >>>     if wav_num == 0:
        >>>         waveforms = np.expand_dims(wav, axis=0)
        >>>     else:
        >>>         wav = np.expand_dims(np.pad(wav, (0, maxlen-wavlen), \
        'constant'), axis=0)
        >>>         waveforms = np.concatenate((waveforms, wav), axis=0)
        >>>     wav_num += 1
        >>> lengths = np.array(lenlist)/maxlen
        >>> dropped_waveform = augment.drop_chunk(waveforms, lengths, \
        drop_start=100, drop_end=200, noise_factor=0.0)


    """

    # Validate low < high
    if drop_length_low > drop_length_high:
        raise ValueError("Low limit must not be more than high limit")
    if drop_count_low > drop_count_high:
        raise ValueError("Low limit must not be more than high limit")

    # Make sure the length doesn't exceed end - start
    if drop_end is not None and drop_end >= 0:
        if drop_start > drop_end:
            raise ValueError("Low limit must not be more than high limit")

        drop_range = drop_end - drop_start
        drop_length_low = min(drop_length_low, drop_range)
        drop_length_high = min(drop_length_high, drop_range)

    # Reading input list
    lengths = lengths * waveforms.shape[1]
    batch_size = waveforms.shape[0]
    dropped_waveform = waveforms.copy()

    # Don't drop (return early) 1-`drop_prob` portion of the batches
    if np.random.rand(1) > drop_prob:
        return dropped_waveform

    # Store original amplitude for computing white noise amplitude
    clean_amplitude = compute_amplitude(waveforms, np.expand_dims(lengths, axis=1))

    # Pick a number of times to drop
    drop_times = np.random.randint(
        low=drop_count_low, high=drop_count_high + 1, size=(batch_size,),
    )

    # Iterate batch to set mask
    for i in range(batch_size):
        if drop_times[i] == 0:
            continue

        # Pick lengths
        length = np.random.randint(
            low=drop_length_low, high=drop_length_high + 1, size=(drop_times[i],),
        )

        # Compute range of starting locations
        start_min = drop_start
        if start_min < 0:
            start_min += lengths[i]
        start_max = drop_end
        if start_max is None:
            start_max = lengths[i]
        if start_max < 0:
            start_max += lengths[i]
        start_max = max(0, start_max - length.max())

        # Pick starting locations
        start = np.random.randint(
            low=start_min, high=start_max + 1, size=(drop_times[i],),
        )

        end = start + length

        # Update waveform
        if not noise_factor:
            for j in range(drop_times[i]):
                dropped_waveform[i, start[j] : end[j]] = 0.0
        else:
            # Uniform distribution of -2 to +2 * avg amplitude should
            # preserve the average for normalization
            noise_max = 2 * clean_amplitude[i] * noise_factor
            for j in range(drop_times[i]):
                # zero-center the noise distribution
                noise_vec = np.random.rand(length[j])
                noise_vec = 2 * noise_max * noise_vec - noise_max
                dropped_waveform[i, start[j] : end[j]] = noise_vec

    return dropped_waveform


def time_stretch(waveforms, rate=None) -> np.ndarray:
    """
    Time-stretch an audio series by a fixed rate.
    Stretch Short Time Fourier Transform (STFT) in time without modifying
    pitch for a given rate.

    Args:
        waveforms (np.ndarray): Shape `[batch, time]`or `[time]`
        rate (float): Rate to speed up or slow down by. Default: None,
        will keep the original rate.

    Returns:
        transfomed waveforms(np.ndarray): Shape `[batch, time]`

    Example:
        >>> signal, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> y_fast = augment.time_stretch(signal, rate=2.0)
    """
    if rate <= 0:
        raise ValueError("rate must be a positive number")

    # Construct the short-term Fourier transform (STFT)
    spec = stft(waveforms)

    # Stretch with the function phase_vocoder
    spec_stretch = _phase_vocoder(spec, rate=rate)
    length_stretch = int(round(waveforms.shape[-1] / rate))

    # Invert to wav
    wav_stretch = istft(spec_stretch, length=length_stretch)
    return wav_stretch


def _phase_vocoder(matrix, rate, hop_length=None, n_fft=None):
    """
    .. [#] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/
    .. [#] https://breakfastquay.com/rubberband/
    """
    if n_fft is None:
        n_fft = 2 * (matrix.shape[-2] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, matrix.shape[-1], rate, dtype=np.float64)

    # Create an empty output array
    shape = list(matrix.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(matrix, shape=shape)

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, matrix.shape[-2])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(matrix[..., 0])

    # Pad 0 columns to simplify boundary logic
    padding = [(0, 0) for _ in matrix.shape]
    padding[-1] = (0, 2)
    matrix = np.pad(matrix, padding, mode="constant")

    for t, step in enumerate(time_steps):
        columns = matrix[..., int(step) : int(step + 2)]
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])
        phase_complex = np.cos(phase_acc) + 1j * np.sin(phase_acc)
        if mag is not None:
            phase_complex *= mag
        d_stretch[..., t] = phase_complex
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase

    return d_stretch


def pitch_shift(waveforms, sr, n_steps, bins_per_octave=12):
    """
    Shift the waveform pitch with n_steps

    Args:
        waveforms: np.ndarray Shape `[batch, time]`or `[time]` audio
        time series.
        sr(int): audio sampling rate
        n_steps(float): steps(fractional) to shift
        bins_per_octave(float): steps per octave

    Returns:
         transfomed waveforms(np.ndarray): Shape `[batch, time]`

    Example:
        >>> waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
        >>> shift_waveform = augment.pitch_shift(waveform, sr=16000, n_steps=4)

    """
    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    waveforms_stretch = time_stretch(waveforms, rate=rate)
    # Stretch in time, then resample
    y_shift = resample(waveforms_stretch, orig_freq=float(sr) / rate, new_freq=sr,)
    return _pad_shape(y_shift, data_shape=waveforms_stretch.shape[-1])
