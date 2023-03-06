import mindspore as ms
import os
import wget
import zipfile
import mindaudio.data.io as io
from mindaudio.data.augment import add_reverb, add_babble, add_noise, drop_freq, speed_perturb, drop_chunk

OPENRIR_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"


class AddNoise:
    """This class additively combines a noise signal to the input signal.
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
        noisy_waveform = add_noise(waveforms, self.noise_data, self.snr_low, self.snr_high, self.mix_prob)

        # Normalizing to prevent clipping
        if self.normalize:
            abs_max, _ = ms.ops.max(
                ms.ops.abs(noisy_waveform), axis=1, keep_dims=True
            )
            noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

        return noisy_waveform


class AddReverb:
    # This class convolve an audio signal with an impulse response.

    def __init__(
            self,
            csv_file,
            reverb_prob=1.0,
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
        babbled_waveform = add_babble(waveforms, lengths, self.speaker_count, self.snr_low, self.snr_high,
                                      self.mix_prob)
        return babbled_waveform


class EnvCorrupt:
    """Environmental Corruptions for speech signals: noise, reverb, babble.

    Arguments
    ---------
    reverb_prob : float from 0 to 1
        The probability that each batch will have reverberation applied.
    babble_prob : float from 0 to 1
        The probability that each batch will have babble added.
    noise_prob : float from 0 to 1
        The probability that each batch will have noise added.
    openrir_folder : str
        If provided, download and prepare openrir to this location. The
        reverberation csv and noise csv will come from here unless overridden
        by the ``reverb_csv`` or ``noise_csv`` arguments.
    openrir_max_noise_len : float
        The maximum length in seconds for a noise segment from openrir. Only
        takes effect if ``openrir_folder`` is used for noises. Cuts longer
        noises into segments equal to or less than this length.
    reverb_csv : str
        A prepared csv file for loading room impulse responses.
    noise_csv : str
        A prepared csv file for loading noise data.
    noise_num_workers : int
        Number of workers to use for loading noises.
    babble_speaker_count : int
        Number of speakers to use for babble. Must be less than batch size.
    babble_snr_low : int
        Lowest generated SNR of reverbed signal to babble.
    babble_snr_high : int
        Highest generated SNR of reverbed signal to babble.
    noise_snr_low : int
        Lowest generated SNR of babbled signal to noise.
    noise_snr_high : int
        Highest generated SNR of babbled signal to noise.
    """

    def __init__(
            self,
            reverb_prob=1.0,
            babble_prob=1.0,
            noise_prob=1.0,
            openrir_folder=None,
            openrir_max_noise_len=None,
            reverb_csv=None,
            noise_csv=None,
            noise_num_workers=0,
            babble_speaker_count=0,
            babble_snr_low=0,
            babble_snr_high=0,
            noise_snr_low=0,
            noise_snr_high=0,
    ):
        # Download and prepare openrir
        if openrir_folder and (not reverb_csv or not noise_csv):

            open_reverb_csv = os.path.join(openrir_folder, "reverb.csv")
            open_noise_csv = os.path.join(openrir_folder, "noise.csv")
            _prepare_openrir(
                openrir_folder,
                open_reverb_csv,
                open_noise_csv,
                openrir_max_noise_len,
            )

            # Specify filepath and sample rate if not specified already
            if not reverb_csv:
                reverb_csv = open_reverb_csv

            if not noise_csv:
                noise_csv = open_noise_csv

        # Initialize corrupters
        if reverb_csv is not None and reverb_prob > 0.0:
            self.add_reverb = AddReverb(
                reverb_prob=reverb_prob,
                csv_file=reverb_csv,
            )

        if babble_speaker_count > 0 and babble_prob > 0.0:
            self.add_babble = AddBabble(
                mix_prob=babble_prob,
                speaker_count=babble_speaker_count,
                snr_low=babble_snr_low,
                snr_high=babble_snr_high,
            )

        if noise_csv is not None and noise_prob > 0.0:
            self.add_noise = AddNoise(
                mix_prob=noise_prob,
                csv_file=noise_csv,
                num_workers=noise_num_workers,
                snr_low=noise_snr_low,
                snr_high=noise_snr_high,
            )

    def construct(self, waveforms, lengths):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort.
        """
        # Augmentation

        if hasattr(self, "add_reverb"):
            waveforms = self.add_reverb.construct(waveforms)
        if hasattr(self, "add_babble"):
            waveforms = self.add_babble.construct(waveforms, lengths)
        if hasattr(self, "add_noise"):
            waveforms = self.add_noise.construct(waveforms)

        return waveforms


def _prepare_openrir(folder, reverb_csv, noise_csv, max_noise_len):
    """Prepare the openrir dataset for adding reverb and noises.

    Arguments
    ---------
    folder : str
        The location of the folder containing the dataset.
    reverb_csv : str
        Filename for storing the prepared reverb csv.
    noise_csv : str
        Filename for storing the prepared noise csv.
    max_noise_len : float
        The maximum noise length in seconds. Noises longer
        than this will be cut into pieces.
    """

    # Download and unpack if necessary
    filepath = os.path.join(folder, "rirs_noises.zip")

    if not os.path.exists(filepath):
        wget.download(OPENRIR_URL, filepath)

    rirs_path_dir = os.path.join(folder, "RIRS_NOISES")
    if not os.path.isdir(rirs_path_dir):
        file = zipfile.ZipFile(filepath)
        os.makedirs(rirs_path_dir)
        file.extractall(rirs_path_dir)
        file.close()

    # Prepare reverb csv if necessary
    if not os.path.isfile(reverb_csv):
        rir_filelist = os.path.join(
            folder, "RIRS_NOISES", "real_rirs_isotropic_noises", "rir_list"
        )
        _prepare_csv(folder, rir_filelist, reverb_csv)

    # Prepare noise csv if necessary
    if not os.path.isfile(noise_csv):
        noise_filelist = os.path.join(
            folder, "RIRS_NOISES", "pointsource_noises", "noise_list"
        )
        _prepare_csv(folder, noise_filelist, noise_csv, max_noise_len)


def _prepare_csv(folder, filelist, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csv_file : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """
    with open(csv_file, "w") as w:
        w.write("ID,duration,wav,wav_format,wav_opts\n\n")
        for line in open(filelist):

            # Read file for duration/channel info
            filename = os.path.join(folder, line.split()[-1])
            signal, rate = io.read(filename)

            # Ensure only one channel
            if signal.shape[0] > 1:
                signal = signal[0].unsqueeze(0)
                io.read(filename, signal, rate)

            wav_id, ext = os.path.basename(filename).split(".")
            duration = signal.shape[1] / rate

            # Handle long waveforms
            if max_length is not None and duration > max_length:
                # Delete old file
                os.remove(filename)
                for i in range(int(duration / max_length)):
                    start = int(max_length * i * rate)
                    stop = int(
                        min(max_length * (i + 1), duration) * rate
                    )
                    new_filename = (
                            filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                    )
                    io.read(
                        new_filename, signal[:, start:stop], rate
                    )
                    csv_row = (
                        f"{wav_id}_{i}",
                        str((stop - start) / rate),
                        new_filename,
                        ext,
                        "\n",
                    )
                    w.write(",".join(csv_row))
            else:
                w.write(
                    ",".join((wav_id, str(duration), filename, ext, "\n"))
                )


class TimeDomainSpecAugment:
    """A time-domain approximation of the SpecAugment algorithm.

    This augmentation module implements three augmentations in
    the time-domain.

     1. Drop chunks of the audio (zero amplitude or white noise)
     2. Drop frequency bands (with band-drop filters)
     3. Speed peturbation (via resampling to slightly different rate)

    Arguments
    ---------
    perturb_prob : float from 0 to 1
        The probability that a batch will have speed perturbation applied.
    drop_freq_prob : float from 0 to 1
        The probability that a batch will have frequencies dropped.
    drop_chunk_prob : float from 0 to 1
        The probability that a batch will have chunks dropped.
    speeds : list of ints
        A set of different speeds to use to perturb each batch.
        See ``speechbrain.processing.speech_augmentation.SpeedPerturb``
    sample_rate : int
        Sampling rate of the input waveforms.
    drop_freq_count_low : int
        Lowest number of frequencies that could be dropped.
    drop_freq_count_high : int
        Highest number of frequencies that could be dropped.
    drop_chunk_count_low : int
        Lowest number of chunks that could be dropped.
    drop_chunk_count_high : int
        Highest number of chunks that could be dropped.
    drop_chunk_length_low : int
        Lowest length of chunks that could be dropped.
    drop_chunk_length_high : int
        Highest length of chunks that could be dropped.
    drop_chunk_noise_factor : float
        The noise factor used to scale the white noise inserted, relative to
        the average amplitude of the utterance. Default 0 (no noise inserted).

    Example
    -------
    >>> inputs = torch.randn([10, 16000])
    >>> feature_maker = TimeDomainSpecAugment(speeds=[80])
    >>> feats = feature_maker(inputs, torch.ones(10))
    >>> feats.shape
    torch.Size([10, 12800])
    """

    def __init__(
            self,
            perturb_prob=1.0,
            drop_freq_prob=1.0,
            drop_chunk_prob=1.0,
            speeds=[95, 100, 105],
            sample_rate=16000,
            drop_freq_count_low=0,
            drop_freq_count_high=3,
            drop_chunk_count_low=0,
            drop_chunk_count_high=5,
            drop_chunk_length_low=1000,
            drop_chunk_length_high=2000,
            drop_chunk_noise_factor=0,
    ):
        self.sample_rate = sample_rate
        self.perturb_prob = perturb_prob
        self.speeds = speeds
        self.drop_freq_prob = drop_freq_prob
        self.drop_freq_count_low = drop_freq_count_low
        self.drop_freq_count_high = drop_freq_count_high
        self.drop_chunk_prob = drop_chunk_prob
        self.drop_chunk_count_low = drop_chunk_count_low
        self.drop_chunk_count_high = drop_chunk_count_high
        self.drop_chunk_length_low = drop_chunk_length_low
        self.drop_chunk_length_high = drop_chunk_length_high
        self.drop_chunk_noise_factor = drop_chunk_noise_factor

    def construct(self, waveforms, lengths):
        """Returns the distorted waveforms.

        Arguments
        ---------
        waveforms : torch.Tensor
            The waveforms to distort
        """
        # Augmentation
        waveforms = speed_perturb(waveforms, self.sample_rate, self.speeds)
        waveforms = drop_freq(waveforms)
        waveforms = drop_chunk(waveforms, lengths)

        return waveforms
