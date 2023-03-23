import os
import zipfile

import mindspore as ms
import numpy as np
import wget

import mindaudio.data.io as io
from mindaudio.data.augment import add_babble, add_noise, add_reverb, drop_chunk, drop_freq, speed_perturb
from mindaudio.data.processing import stereo_to_mono

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"


class InputNormalization:
    """
    Performs mean and variance normalization of the input tensor.
    Args:
        mean_norm : True, If True, the mean will be normalized.
        std_norm : True, If True, the standard deviation will be normalized.
        norm_type : str, 'sentence' computes them at sentence level, 'batch'
            at batch level, 'speaker' at speaker level.
    """

    def __init__(
        self, mean_norm=True, std_norm=True, norm_type="global",
    ):
        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.eps = 1e-10

    def construct(self, x_input):
        batches = x_input.shape[0]
        for snt_id in range(batches):

            # Avoiding padded time steps
            actual_size = x_input.shape[1]

            current_mean, current_std = self._compute_current_stats(x_input[snt_id, 0:actual_size, ...])

            if self.norm_type == "sentence":
                x_input[snt_id] = (x_input[snt_id] - current_mean) / current_std

        return x_input

    def _compute_current_stats(self, x_input):
        if self.mean_norm:
            current_mean = np.mean(x_input, axis=0)
        else:
            current_mean = np.array([0.0])

        # Compute current std
        if self.std_norm:
            current_std = np.std(x_input, axis=0)
        else:
            current_std = np.array([1.0])

        return current_mean, current_std


class AddNoise:
    """
    This class additively combines a noise signal to the input signal.
    """

    def __init__(
        self,
        csv_file=None,
        csv_keys=None,
        sorting="random",
        num_workers=0,
        snr_low=0,
        snr_high=0,
        pad_noise=False,
        mix_prob=1.0,
        start_index=None,
        normalize=False,
    ):
        self.csv_file = csv_file
        self.csv_keys = csv_keys
        self.sorting = sorting
        self.num_workers = num_workers
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.pad_noise = pad_noise
        self.mix_prob = mix_prob
        self.start_index = start_index
        self.normalize = normalize
        self.noise_data = []
        noise_dataset = ms.dataset.CSVDataset(dataset_files=self.csv_file, shuffle=True)
        noise_dataset = noise_dataset.project(columns=["wav"])

        iterator = noise_dataset.create_dict_iterator(num_epochs=1)

        for batch in iterator:
            wav_file = batch["wav"]
            self.noise_data.append(str(wav_file))

    def construct(self, waveforms):
        noisy_waveform = add_noise(waveforms, self.noise_data, self.snr_low, self.snr_high, self.mix_prob,)

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = ms.ops.max(ms.ops.abs(noisy_waveform), axis=1, keep_dims=True)
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform


class AddReverb:
    # This class convolve an audio signal with an impulse response.

    def __init__(
        self, csv_file, reverb_prob=1.0,
    ):
        self.csv_file = csv_file
        self.reverb_prob = reverb_prob
        self.rir_data = []
        rir_dataset = ms.dataset.CSVDataset(dataset_files=self.csv_file, shuffle=True)
        rir_dataset = rir_dataset.project(columns=["wav"])

        iterator = rir_dataset.create_dict_iterator(num_epochs=1)

        for batch in iterator:
            wav_file = batch["wav"]
            self.rir_data.append(str(wav_file))

    def construct(self, waveforms):
        rev_waveform = add_reverb(waveforms, self.rir_data, self.reverb_prob)
        return rev_waveform


class AddBabble:
    # Simulate babble noise by mixing the signals in a batch.

    def __init__(
        self, speaker_count=3, snr_low=0, snr_high=0, mix_prob=1,
    ):
        self.speaker_count = speaker_count
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.mix_prob = mix_prob

    def construct(self, waveforms, lengths):
        babbled_waveform = add_babble(
            waveforms, lengths, self.speaker_count, self.snr_low, self.snr_high, self.mix_prob,
        )
        return babbled_waveform


class EnvCorrupt:
    """
    Environmental Corruptions for speech signals: noise, reverb, babble.
    """

    def __init__(
        self,
        openrir_folder=None,
        openrir_max_noise_len=None,
        reverb_csv=None,
        noise_csv=None,
        reverb_prob=1.0,
        babble_prob=1.0,
        noise_prob=1.0,
        noise_num_workers=0,
        noise_snr_low=0,
        noise_snr_high=0,
        babble_speaker_count=0,
        babble_snr_low=0,
        babble_snr_high=0,
    ):
        # Download and prepare openrir
        if openrir_folder and (not noise_csv or not reverb_csv):
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")
            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            prepare_openrir(
                openrir_folder, open_reverb_csv, open_noise_csv, openrir_max_noise_len,
            )

            # Specify filepath and sample rate if not specified already
            if not noise_csv:
                noise_csv = open_noise_csv

            if not reverb_csv:
                reverb_csv = open_reverb_csv

        # Initialize corrupters
        if noise_prob > 0.0 and noise_csv is not None:
            self.add_noise = AddNoise(
                csv_file=noise_csv,
                mix_prob=noise_prob,
                num_workers=noise_num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
            )

        if babble_prob > 0.0 and babble_speaker_count > 0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if reverb_prob > 0.0 and reverb_csv is not None:
            self.add_reverb = AddReverb(reverb_prob=reverb_prob, csv_file=reverb_csv,)

    def construct(self, waves, lens):
        """
        Returns the distorted waveforms.

        Args:
            waves : np array, The waveforms to distort.
            lens : int, comparing to max waveform.
        """

        if hasattr(self, "add_reverb"):
            waves = self.add_reverb.construct(waves)
        if hasattr(self, "add_babble"):
            waves = self.add_babble.construct(waves, lens)
        if hasattr(self, "add_noise"):
            waves = self.add_noise.construct(waves)

        return waves


def prepare_openrir(folder, reverb_csv, noise_csv, max_noise_len):
    """
    Prepare the openrir dataset for adding reverb and noises.

    Args:
        folder : str, The location of the folder containing the dataset.
        reverb_csv : str, Filename for storing the prepared reverb csv.
        noise_csv : str, Filename for storing the prepared noise csv.
        max_noise_len : float, The maximum noise length in seconds.
    """

    # Download and unpack if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")

    if not os.path.exists(filepath):
        wget.download(OPENRIR_URL, filepath)

    if not os.path.isdir(os.path.join(folder, "RIRS_NOISES")):
        file = zipfile.ZipFile(filepath)
        file.extractall(folder)
        file.close()

    # Prepare noise csv if necessary
    if not os.path.isfile(noise_csv):
        noise_filelist = os.path.join(folder, "RIRS_NOISES", "pointsource_noises", "noise_list")
        prepare_csv(folder, noise_filelist, noise_csv, max_noise_len)

    # Prepare reverb csv if necessary
    if not os.path.isfile(reverb_csv):
        rir_filelist = os.path.join(folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list")
        prepare_csv(folder, rir_filelist, reverb_csv)


def prepare_csv(folder, filelist, csv_file, max_length=None):
    """
    Iterate a set of wavs and write the corresponding csv file.

    Args:
        folder : str, The folder relative to which the files in the list.
        filelist : str, The location of a file listing the files to be used.
        csv_file : str, The location to use for writing the csv file.
        max_length : float, The maximum length in seconds.
    """

    with open(csv_file, "w") as w_csv_file:
        w_csv_file.write("ID,duration,wav,wav_format,wav_opts\n\n")
        for line in open(filelist):

            # Read file for duration/channel info
            filename = os.path.join(folder, line.split()[-1])
            signal, rate = io.read(filename)

            # Ensure only one channel
            if len(signal.shape) > 1:
                signal = stereo_to_mono(signal)
                io.write(filename, signal, rate)

            wav_id, ext = os.path.basename(filename).split(".")
            duration = signal.shape[0] / rate

            # Handle long waveforms
            if max_length is not None and duration > max_length:
                # Delete old file
                os.remove(filename)
                for index in range(int(duration / max_length)):
                    start = int(max_length * index * rate)
                    stop = int(min(max_length * (index + 1), duration) * rate)
                    new_filename = filename[: -len(f".{ext}")] + f"_{index}.{ext}"
                    io.write(new_filename, signal[start:stop], rate)
                    csv_row = (
                        f"{wav_id}_{index}",
                        str((stop - start) / rate),
                        new_filename,
                        ext,
                        "\n",
                    )
                    w_csv_file.write(",".join(csv_row))
            else:
                w_csv_file.write(",".join((wav_id, str(duration), filename, ext, "\n")))


class TimeDomainSpecAugment:
    """
    A time-domain approximation of the SpecAugment algorithm.

    This augmentation module implements three augmentations in
    the time-domain.
     1. Drop chunks of the audio
     2. Drop frequency bands
     3. Speed peturbation
    """

    def __init__(
        self,
        speeds=[95, 100, 105],
        sample_rate=16000,
        perturb_prob=1.0,
        drop_freq_prob=1.0,
        drop_chunk_prob=1.0,
        drop_chunk_length_low=1000,
        drop_chunk_length_high=2000,
        drop_chunk_count_low=0,
        drop_chunk_count_high=5,
        drop_freq_count_low=0,
        drop_freq_count_high=3,
        drop_chunk_noise_factor=0,
    ):
        self.speeds = speeds
        self.sample_rate = sample_rate
        self.perturb_prob = perturb_prob
        self.drop_chunk_count_low = drop_chunk_count_low
        self.drop_chunk_count_high = drop_chunk_count_high
        self.drop_chunk_length_low = drop_chunk_length_low
        self.drop_chunk_length_high = drop_chunk_length_high
        self.drop_chunk_noise_factor = drop_chunk_noise_factor
        self.drop_freq_prob = drop_freq_prob
        self.drop_freq_count_low = drop_freq_count_low
        self.drop_freq_count_high = drop_freq_count_high
        self.drop_chunk_prob = drop_chunk_prob

    def construct(self, waves, lens):
        """
        Returns the distorted waveforms.

        Args:
            waves : np array, The waveforms to distort
        """

        waves = speed_perturb(waves, self.sample_rate, self.speeds)
        waves = drop_freq(waves)
        waves = drop_chunk(waves, lens)

        return waves
