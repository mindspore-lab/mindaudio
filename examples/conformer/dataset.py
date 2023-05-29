"""ASR Training Data Generator."""

import codecs
import math
import multiprocessing as mp
import os
import random
from multiprocessing import Pool

import mindspore.dataset.engine as de
import numpy as np

import mindaudio
from mindaudio.utils.common import IGNORE_ID, add_sos_eos, pad_sequence
from mindaudio.utils.log import get_logger
from mindaudio.utils.mask import (add_optional_chunk_mask, make_pad_mask,
                                  subsequent_mask)

logger = get_logger()


def get_waveform(batch, sample_rate=16000):
    """Load audios.

    Args:
        batch (list): a list of (uttid, wav_path, labels)
        sample_rate (int): sample rate of audios. Defaults to 16000.

    Return:
        tuple: (uttids, wav_samples, wav_lengths, labels)
    """
    uttids = []
    wavs = []
    lengths = []
    for _, x in enumerate(batch):
        wav_path = x[1]
        waveform, sample_rate = mindaudio.read(wav_path)
        waveform = waveform * (1 << 15)
        uttids.append(x[0])
        wavs.append(waveform)
        lengths.append(waveform.shape[0])

    # Sort it because sorting is required in pack/pad operation
    order = np.argsort(lengths)[::-1]
    sorted_uttids = [uttids[i] for i in order]
    sorted_wavs = [wavs[i] for i in order]
    sorted_lengths = [lengths[i] for i in order]
    labels = [x[2].split() for x in batch]
    labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
    sorted_labels = [labels[i] for i in order]

    return sorted_uttids, sorted_wavs, sorted_lengths, sorted_labels


def inverse_mel_scale(mel_freq):
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def mel_scale(freq):
    return 1127.0 * np.log(1.0 + freq / 700.0)


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
):
    """Get mel banks for extracting features.

    Args:
        num_bins (int): number of mel bins.
        window_length_padded (int): length of windows to split frame.
        sample_freq (int): sample rate of audios.
        low_freq (float): lowest frequency.
        high_freq (float): highest frequency.
    """

    num_fft_bins = window_length_padded // 2

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    bins = np.arange(num_bins).reshape(-1, 1)
    left_mel = mel_low_freq + bins * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bins + 1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bins + 2.0) * mel_freq_delta  # size(num_bins, 1)

    center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    # mel = mel_scale(fft_bin_width * np.arange(num_fft_bins)).unsqueeze(0)
    mel = np.expand_dims(mel_scale(fft_bin_width * np.arange(num_fft_bins)), 0)
    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
    feat = np.where(up_slope > down_slope, down_slope, up_slope)
    feat = np.where(feat < 0, 0, feat)
    feat = np.pad(feat, ((0, 0), (0, 1)), "constant")

    return feat, center_freqs


# Enframe with Hamming window function
def preemphasis(signal):
    """Perform preemphasis on the input signal."""
    return np.append(signal[0], signal[1:] - 0.97 * signal[:-1])


def enframe(signal, frame_len, frame_shift):
    """Enframe with Hamming widow function."""

    num_samples = signal.size
    win = np.power(np.hanning(frame_len), 0.85)
    num_frames = np.floor((num_samples - frame_len) / frame_shift) + 1
    frames = np.zeros((int(num_frames), frame_len))
    for i in range(int(num_frames)):
        frames[i, :] = signal[i * frame_shift : i * frame_shift + frame_len]
        frames[i, :] = frames[i, :] * win
    return frames


def get_spectrum(frames, fft_len):
    """Get spectrum using fft."""
    c_fft = np.fft.rfft(frames, n=fft_len)
    spectrum = np.abs(c_fft) ** 2
    return spectrum


def fbank(spectrum, num_filter, fs):
    """Get mel filter bank features from spectrum.

    args:
        spectrum: a num_frames by fft_len/2+1 array(real)
        num_filter: mel filters number, default 23

    return:
        fbank feature, a num_frames by num_filter array
    """
    mel_energies = get_mel_banks(num_filter, 512, fs * 2, 20, 8000)[0]
    feats = np.dot(spectrum, mel_energies.T)
    feats = np.where(feats == 0, np.finfo(float).eps, feats)
    feats = np.log(feats)
    return feats


def compute_fbank_feats(wav, sample_rate, frame_len, frame_shift, mel_bin):
    """compute fbank feats."""
    signal = preemphasis(wav)
    frame_len = sample_rate * frame_len // 1000
    frame_shift = sample_rate * frame_shift // 1000
    frames = enframe(signal, frame_len=frame_len, frame_shift=frame_shift)
    frames -= np.mean(frames)
    spectrum = get_spectrum(frames, fft_len=512)
    fbank_feats = fbank(spectrum, num_filter=mel_bin, fs=sample_rate / 2)
    return fbank_feats


def get_padding_length(length, frame_bucket_limits):
    """Get the padding length through th bucket limitation."""
    for limit in frame_bucket_limits:
        if limit > length:
            return limit
    return frame_bucket_limits[-1]


def safe_readline(f):
    """Safety read line."""
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)


def load_samples(data_file, worker_id, frame_factor, workers_num):
    """Load all training samples from data file."""
    data = []
    with codecs.open(data_file, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // workers_num
        offset = worker_id * chunk_size
        end = offset + chunk_size
        f.seek(offset)
        logger.info("offset %d", offset)
        # TODO: whether need safe readline
        if offset > 0:
            safe_readline(f)  # drop first incomplete line
        line = f.readline()
        miss_file_cnt = 0
        while line:
            arr = line.strip().split("\t")
            # len(arr) == 7 for ASR and wav2vec 2.0 training
            # len(arr) == 8 for Hubert training
            if len(arr) != 7 and len(arr) != 8:
                line = f.readline()
                continue
            uttid = arr[0].split(":")[1]
            tokenid = arr[5].split(":")[1]
            output_dim = int(arr[6].split(":")[1].split(",")[1])
            wav_path = ":".join(arr[1].split(":")[1:])
            duration = int(float(arr[2].split(":")[1]) * frame_factor)
            if not os.path.exists(wav_path):
                miss_file_cnt += 1
                line = f.readline()
                continue
            if len(arr) == 7:
                data.append((uttid, wav_path, duration, tokenid, output_dim))
            if len(arr) == 8:
                kmeans_id = arr[7].split(":")[1]
                data.append((uttid, wav_path, duration, tokenid, output_dim, kmeans_id))
            if f.tell() > end:
                break
            line = f.readline()
        logger.info("Missing file num: %d", miss_file_cnt)
        return data


def parse_file(path, frame_factor, workers=8):
    """parse index file."""
    assert os.path.exists(path)
    results = []
    workers_thread = []
    pool = Pool(processes=workers)
    for i in range(workers):
        w = pool.apply_async(load_samples, args=(path, i, frame_factor, workers))
        workers_thread.append(w)
    pool.close()
    pool.join()
    for w in workers_thread:
        result = w.get()
        results += result
    return results


class BucketDatasetBase:
    """Create BucketDatasetBase.

    Args:
        data_file (str): input data file.
        frame_bucket_limit (str): input length limitation of each bucket.
        batch_bucket_limit (str): batch size (number of sample) of each bucket.
        batch_factor (float): scale factor to batch_bucket_limit, the final.
                              batch is decided by batch_bucket_limit * batch_factor.
        frame_factor (int): frame factor.
        group_size (int): number of total world size, for multi-GPUs distributed training.
    """

    def __init__(
        self,
        data_file,
        frame_bucket_limit="200,300",
        batch_bucket_limit="220,200",
        batch_factor=0.2,
        frame_factor=100,
        group_size=1,
    ):
        self.group_size = group_size
        self.frame_bucket_limit = [int(i) for i in frame_bucket_limit.split(",")]
        self.batch_bucket_limit = [
            int(int(i) * batch_factor * group_size)
            for i in batch_bucket_limit.split(",")
        ]
        assert len(self.frame_bucket_limit) == len(self.batch_bucket_limit)
        self.bucket_select_dict = self.bucket_init(self.frame_bucket_limit)

        # load all samples
        data = parse_file(data_file, frame_factor, workers=1)
        # sort all date according to their lengths
        # each item of data include [uttid, wav_path, duration, tokenid, output_dim, kmeans_id (optional)]
        self.data = sorted(data, key=lambda x: x[2])
        # implement by subclass, each item includes: [data, max_limit_frame]
        self.batches = []

    def bucket_init(self, frame_bucket_limit):
        """Init a bucket selection dict based on frame_bucket_limit."""
        bucket_select_dict = {}
        for idx, _ in enumerate(frame_bucket_limit):
            low = 0 if idx == 0 else frame_bucket_limit[idx - 1] + 1
            high = frame_bucket_limit[idx] + 1
            bucket_select_dict.update({i: idx for i in range(low, high)})

        return bucket_select_dict

    def __getitem__(self, index):
        return self.batches[index][0], self.batches[index][1]

    def __len__(self):
        return len(self.batches)


class BucketASRDataset(BucketDatasetBase):
    """Create BucketASRDataset.

    Args:
        data_file (str): input data file.
        max_length (int): maximum length of input audio file.
        min_length (int): minimum length of input audio file.
        token_max_length (int): maximum length of label.
        token_min_length (int): minimum length of label.
        frame_bucket_limit (str): input length limitation of each bucket.
        batch_bucket_limit (str): batch size (number of sample) of each bucket.
        batch_factor (float): scale factor to batch_bucket_limit, the final.
                              batch is decided by batch_bucket_limit * batch_factor.
        frame_factor (int): frame factor.
        group_size (int): number of total world size, for multi-GPUs distributed training.
    """

    def __init__(
        self,
        data_file,
        max_length=10240,
        min_length=0,
        token_max_length=200,
        token_min_length=1,
        frame_bucket_limit="200,300",
        batch_bucket_limit="220,200",
        batch_factor=0.2,
        frame_factor=100,
        group_size=1,
    ):
        super().__init__(
            data_file,
            frame_bucket_limit=frame_bucket_limit,
            batch_bucket_limit=batch_bucket_limit,
            batch_factor=batch_factor,
            frame_factor=frame_factor,
            group_size=group_size,
        )
        self.token_max_length = token_max_length
        # load all samples
        num_sample = 0
        tot_num_sample = len(self.data)
        self.batches = []
        caches = {}  # caches to store data
        for idx, max_frame in enumerate(self.frame_bucket_limit):
            # caches[idx]: [data, num_sentence, max_frame]
            caches[idx] = [[], 0, max_frame]
        for i in range(len(self.data)):
            length = self.data[i][2]
            token_length = len(self.data[i][3].split())
            self.output_dim = self.data[i][4]
            # remove too lang or too short utt for both input and output
            if length > max_length or length < min_length:
                continue
            elif token_length > token_max_length or token_length < token_min_length:
                continue
            else:
                num_sample += 1
                bucket_idx = self.bucket_select_dict[length]
                caches[bucket_idx][0].append(
                    (self.data[i][0], self.data[i][1], self.data[i][3])
                )
                caches[bucket_idx][1] += 1

                if caches[bucket_idx][1] >= self.batch_bucket_limit[bucket_idx]:
                    self.batches.append((caches[bucket_idx][0], caches[bucket_idx][2]))
                    caches[bucket_idx] = [[], 0, self.frame_bucket_limit[bucket_idx]]

        # handle the left samples which are not able to form a complete batch
        for key, value in caches.items():
            length = len(value[0])
            if length != 0:
                repeat_time = math.ceil(self.batch_bucket_limit[key] / length)
                data_expand = value[0] * repeat_time
                self.batches.append(
                    (data_expand[: self.batch_bucket_limit[key]], value[2])
                )
        del caches

        logger.info(
            "Total utts: %d, remove too long/short utts: %d.",
            num_sample,
            tot_num_sample - num_sample,
        )
        self.sos = self.output_dim - 1
        self.eos = self.output_dim - 1

    def __getitem__(self, index):
        data, max_src_len = self.batches[index][0], self.batches[index][1]
        return data, self.sos, self.eos, max_src_len, self.token_max_length


def parallel_compute_fbank_feats(param):
    """compute fbank feats."""
    wav_path = param[0][1]
    use_speed_perturb = param[1]

    waveform, sample_rate = mindaudio.read(wav_path)
    waveform = waveform * (1 << 15)

    if use_speed_perturb:
        waveform = speed_perturb(waveform, sample_rate=16000)

    return compute_fbank_feats(waveform, sample_rate, param[2], param[3], param[4])


def speed_perturb(waveform, sample_rate):
    """speed perturb."""
    speeds = [0.9, 1.0, 1.1]
    speed = random.choice(speeds)

    if speed != 1.0:
        waveform = mindaudio.resample(waveform, sample_rate * speed, sample_rate)

    return waveform


class CollateFunc:
    """Collate function for audio dataset.

    Args:
        rank (int): current rank of the total world size, for multi-GPUs distributed training.
        group_size (int): number of total world size, for multi-GPUs distributed training.
        feature_extraction_conf (dict): configuration for feature extraction.
        feature_dither (float): factor of feature dither.
        speed_perturb (bool): whether to use speed perturbation.
        spec_aug (bool): whether to use specaugment.
        spec_aug_conf (dict): configuration for specaugment.
        use_dynamic_chunk (bool): whether to use dynamic chunk, for streaming ASR.
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk, for streaming ASR.
        decoding_chunk_size (int): number of chunk use for inference, for streaming ASR.
        static_chunk_size (int): chunk size for streaming ASR.
        num_decoding_left_chunks (int):  number of left chunk use for inference, for streaming ASR.
    """

    def __init__(
        self,
        rank,
        group_size,
        feature_extraction_conf=None,
        feature_dither=0.0,
        use_speed_perturb=False,
        use_spec_aug=False,
        spec_aug_conf=None,
        use_dynamic_chunk=False,
        use_dynamic_left_chunk=False,
        decoding_chunk_size=0,
        static_chunk_size=0,
        num_decoding_left_chunks=-1,
    ):
        self.feature_extraction_conf = feature_extraction_conf
        self.feature_dither = feature_dither
        self.use_speed_perturb = use_speed_perturb
        self.use_spec_aug = use_spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.rank = rank
        self.group_size = group_size
        self.pool = mp.Pool(8)
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.decoding_chunk_size = decoding_chunk_size
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def extract_feature(self, batch, use_speed_perturb, feature_extraction_conf):
        """Extract acoustic fbank features from the waveforms.

        Args:
            batch (list): a list of tuple (wav_id, wav_path, wav_label).
            speed_perturb (bool): whether to do speed perturbation.
            feature_extraction_conf (dict): configurations of fbank feature extraction.

        Returns:
            tuple: [sorted_uttid, sorted_feats, sorted_labels]
        """
        uttids = []
        feats = []
        lengths = []

        mel_bin = int(feature_extraction_conf["mel_bins"])
        frame_len = int(feature_extraction_conf["frame_length"])
        frame_shift = int(feature_extraction_conf["frame_shift"])
        param = []
        for _, x in enumerate(batch):
            param.append((x, use_speed_perturb, frame_len, frame_shift, mel_bin))
            uttids.append(x[0])

        res = self.pool.map(parallel_compute_fbank_feats, param)
        for feat in res:
            feats.append(feat)
            lengths.append(feat.shape[0])
        # Sort the batch because sorting is required in pack/pad operation
        order = np.argsort(lengths)[::-1]
        sorted_uttids = [uttids[i] for i in order]
        sorted_feats = [feats[i] for i in order]
        labels = [x[2].split() for x in batch]
        labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
        sorted_labels = [labels[i] for i in order]

        return sorted_uttids, sorted_feats, sorted_labels

    def spec_aug(self, xs, spec_aug_conf):
        """Do specaugment. Inplace operation.

        Args:
            xs: Iterable[{feat}]
            num_t_mask (int): number of time mask to apply
            num_f_mask (int): number of freq mask to apply
            prop_mask_t (int): mask prop of time
            prop_mask_f (int): mask prop of freq
            max_t (int): max width of time mask
            max_f (int): max width of freq mask

        Returns:
            Iterable[{feat}]
        """
        num_t_mask = spec_aug_conf.get("num_t_mask", 0.0)
        num_f_mask = spec_aug_conf.get("num_f_mask", 0.0)
        # prop_mask_t = spec_aug_conf.get('prop_mask_t', 0.0)
        # prop_mask_f = spec_aug_conf.get('prop_mask_f', 0.0)
        max_t = spec_aug_conf.get("max_t", 0.0)
        max_f = spec_aug_conf.get("max_f", 0.0)

        for x in xs:
            max_frames = x.shape[0]
            max_freq = x.shape[1]
            # time mask
            for _ in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                # length = random.randint(1, int(prop_mask_t * max_frames))
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                if random.randint(1, 100) > 20:
                    x[start:end, :] = 0
            # freq mask
            for _ in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                # length = random.randint(1, int(prop_mask_f * max_frames))
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                if random.randint(1, 100) > 20:
                    x[:, start:end] = 0
        return xs

    def __call__(self, batch, sos=0, eos=0, max_src_len=2000, max_tgt_len=30):
        """Feature collate process, including feature extraction, data
        augmentation, feature and label padding, generating mask for feature
        and labels, etc.

        Args:
            batch (list): a list of tuple (wav_id, wav_path, wav_label)
            sos (int): A specific start token padded to labels. Defaults to 0.
            eos (int): A specific end token padded to labels. Defaults to 0.
            max_src_len (int): Maximum feature length. Defaults to 2000.
            max_tgt_len (int): Maximum label length. Defaults to 30.

        Returns:
            tuple: [xs_pad, ys_pad, ys_in_pad, ys_out_pad, xs_masks, ys_sub_mask,
                    ys_masks, ys_lengths, xs_chunk_masks]
        """
        _, xs, ys = self.extract_feature(
            batch[self.rank :: self.group_size],
            self.use_speed_perturb,
            self.feature_extraction_conf,
        )
        if self.feature_dither != 0.0:
            raise NotImplementedError

        if self.use_spec_aug:
            xs = self.spec_aug(xs, self.spec_aug_conf)

        xs_pad = pad_sequence(
            xs,
            batch_first=True,
            padding_value=0.0,
            padding_max_len=max_src_len,
            atype=np.float32,
        )
        ys_pad = pad_sequence(
            ys,
            batch_first=True,
            padding_value=IGNORE_ID,
            padding_max_len=max_tgt_len,
            atype=np.int32,
        )
        # generate the input and output sequence for ASR decoder
        # ys_in: add <sos> label to the ys
        # ys_out: add <eos> label to the ys
        ys_in, ys_out = add_sos_eos(ys, sos, eos)
        # the padding_max_len should be increase by 1, since y is paded with
        # a <sos> or <eos>
        ys_in_pad = pad_sequence(
            ys_in,
            batch_first=True,
            padding_value=eos,
            padding_max_len=max_tgt_len + 1,
            atype=np.int32,
        )
        ys_out_pad = pad_sequence(
            ys_out,
            batch_first=True,
            padding_value=IGNORE_ID,
            padding_max_len=max_tgt_len + 1,
            atype=np.int32,
        )

        # generate the input and output sequence for ASR right-left decoder
        r_ys = [item[::-1] for item in ys]
        r_ys_in, r_ys_out = add_sos_eos(r_ys, sos, eos)
        r_ys_in_pad = pad_sequence(
            r_ys_in,
            batch_first=True,
            padding_value=eos,
            padding_max_len=max_tgt_len + 1,
            atype=np.int32,
        )
        r_ys_out_pad = pad_sequence(
            r_ys_out,
            batch_first=True,
            padding_value=IGNORE_ID,
            padding_max_len=max_tgt_len + 1,
            atype=np.int32,
        )

        xs_lengths = np.array([x.shape[0] for x in xs], dtype=np.int32)
        ys_lengths = np.array([len(y) for y in ys], dtype=np.int32)

        # make xs_masks, (B, 1, T), audio == 1, padding == 0
        xs_masks = np.expand_dims(~make_pad_mask(xs_lengths, max_len=max_src_len), 1)
        xs_masks = xs_masks.astype(np.float32)

        # make ys_masks, (B, 1, T), text == 1, padding == 0
        # the length of each y should be increase by 1, since it is paded with
        # a <sos> or <eos>
        ys_masks = np.expand_dims(
            ~make_pad_mask(ys_lengths + 1, max_len=max_tgt_len + 1), 1
        )
        m = np.expand_dims(subsequent_mask(max_tgt_len + 1), 0)
        ys_sub_masks = (ys_masks & m).astype(np.float32)
        ys_masks = ys_masks.astype(np.float32)
        xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]
        xs_pad_downsample4_len = (xs_pad.shape[1] - 3) // 4
        xs_chunk_masks = add_optional_chunk_mask(
            xs_len=xs_pad_downsample4_len,
            masks=xs_masks,
            use_dynamic_chunk=self.use_dynamic_chunk,
            use_dynamic_left_chunk=self.use_dynamic_left_chunk,
            decoding_chunk_size=self.decoding_chunk_size,
            static_chunk_size=self.static_chunk_size,
            num_decoding_left_chunks=self.num_decoding_left_chunks,
        )

        return (
            xs_pad,
            ys_pad,
            ys_in_pad,
            ys_out_pad,
            r_ys_in_pad,
            r_ys_out_pad,
            xs_masks,
            ys_sub_masks,
            ys_masks,
            ys_lengths,
            xs_chunk_masks,
        )


def create_dataset(
    data_file, collate_conf, dataset_conf, rank=0, group_size=1, number_workers=8
):
    """Init a iterable dataset.

    Args:
        data_file (str): input data file.
        collate_conf (dict): configurations for the collate function.
        dataset_conf (dict): configurations for the dataset.
        rank (int): current rank of the total world size, for multi-GPUs distributed training.
        group_size (int): number of total world size, for multi-GPUs distributed training.
        number_workers (int): number of process workers.

    Returns:
        tuple: the output size and a iterable data generator
    """
    collate_func = CollateFunc(rank=rank, group_size=group_size, **collate_conf)
    dataset = BucketASRDataset(
        data_file,
        max_length=dataset_conf["max_length"],
        min_length=dataset_conf["min_length"],
        token_max_length=dataset_conf["token_max_length"],
        token_min_length=dataset_conf["token_min_length"],
        frame_bucket_limit=dataset_conf["frame_bucket_limit"],
        batch_bucket_limit=dataset_conf["batch_bucket_limit"],
        batch_factor=dataset_conf["batch_factor"],
        frame_factor=100,
        group_size=group_size,
    )

    sampler = DistributedSampler(dataset, rank, group_size, shuffle=True)

    ds = de.GeneratorDataset(
        dataset,
        ["data", "sos", "eos", "max_src_len", "token_max_length"],
        sampler=sampler,
        num_parallel_workers=1,
        max_rowsize=24,
    )
    output_dim = dataset.output_dim

    # pylint: disable=W0108
    map_func = lambda data, sos, eos, max_src_len, token_max_length: collate_func(
        data, sos, eos, max_src_len, token_max_length
    )
    ds = ds.map(
        operations=map_func,
        input_columns=["data", "sos", "eos", "max_src_len", "token_max_length"],
        output_columns=[
            "xs_pad",
            "ys_pad",
            "ys_in_pad",
            "ys_out_pad",
            "r_ys_in_pad",
            "r_ys_out_pad",
            "xs_masks",
            "ys_masks",
            "ys_sub_masks",
            "ys_lengths",
            "xs_chunk_masks",
        ],
        column_order=[
            "xs_pad",
            "ys_pad",
            "ys_in_pad",
            "ys_out_pad",
            "r_ys_in_pad",
            "r_ys_out_pad",
            "xs_masks",
            "ys_masks",
            "ys_sub_masks",
            "ys_lengths",
            "xs_chunk_masks",
        ],
        num_parallel_workers=number_workers,
        python_multiprocessing=False,
    )

    return output_dim, ds


class AsrPredictDataset:
    """Create AsrPredictDataset.

    Args:
        data_file (str): input data file.
        max_length (int): maximum length of input audio file.
        min_length (int): minimum length of input audio file.
        token_max_length (int): maximum length of label.
        token_min_length (int): minimum length of label.
        frame_factor (int): frame factor.
    """

    def __init__(
        self,
        data_file,
        max_length=10240,
        min_length=0,
        token_max_length=200,
        token_min_length=1,
        frame_factor=100,
    ):
        self.token_max_length = token_max_length

        # load all samples
        data = parse_file(data_file, frame_factor, workers=6)
        self.batches = []
        num_sample = 0
        for i in range(len(data)):
            uttid = data[i][0]
            wav_path = data[i][1]
            length = data[i][2]
            tokens = [int(i) for i in data[i][3].split()]
            token_length = len(tokens)

            if length > max_length or length < min_length:
                logger.warning(
                    "Utts %s has %d frames, out of frame limit %d ~ %d, remove it.",
                    data[i][0],
                    length,
                    min_length,
                    max_length,
                )
                continue
            elif token_length > token_max_length or token_length < token_min_length:
                logger.warning(
                    "Utts %s has %d tokens, out of token limit %d ~ %d, remove it.",
                    data[i][0],
                    token_length,
                    token_min_length,
                    token_max_length,
                )
                continue
            else:
                num_sample += 1
                self.batches.append((uttid, wav_path, length, tokens))

        logger.info(
            "Total utts: %d, remove too long/short utts: %d.",
            num_sample,
            len(data) - num_sample,
        )

    def __getitem__(self, index):
        return (
            self.batches[index][0],
            self.batches[index][1],
            self.batches[index][2],
            self.batches[index][3],
        )

    def __len__(self):
        return len(self.batches)


def load_language_dict(dict_file):
    """Load dict for ASR."""
    char_dict = {}
    with open(dict_file, "r") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    sos = len(char_dict) - 1
    eos = len(char_dict) - 1
    vocab_size = len(char_dict)
    return sos, eos, vocab_size, char_dict


def create_e2e_predict_dataset(data_file, extractor_conf, dataset_conf, num_workers=1):
    """Create joint wav2vec 2.0 & ASR predictiong dataset.

    Args:
        data_file (str): input data file.
        extractor_conf (dict): configuration for feature extraction.
        dataset_conf (dict): configurations for the dataset.
        num_workers (int): number of process workers.

    Returns:
        A iterable data generator.
    """
    dataset = AsrPredictDataset(
        data_file,
        dataset_conf["max_length"],
        dataset_conf["min_length"],
        dataset_conf["token_max_length"],
        dataset_conf["token_min_length"],
        16000,
    )

    ds = de.GeneratorDataset(
        dataset,
        ["uutid", "wav_path", "length", "tokens"],
        max_rowsize=12,
        shuffle=False,
    )

    kernel_size = [int(i) for i in extractor_conf["kernel_size_list"].split(",")]
    stride = [int(i) for i in extractor_conf["stride_list"].split(",")]
    frame_bucket_limit = [int(i) for i in dataset_conf["frame_bucket_limit"].split(",")]

    def data_preprocess_e2e(uutid, wav_path, tokens):
        # load wav data
        waveform, _ = mindaudio.read(wav_path.item(0))
        waveform = waveform * (1 << 15)
        xs = waveform
        xs_lengths = waveform.shape[0]

        # pad wavs
        padding_length = get_padding_length(xs_lengths, frame_bucket_limit)
        xs_pad = pad_sequence(
            [xs],
            padding_max_len=padding_length,
            batch_first=True,
            padding_value=0,
            atype=np.float32,
        )

        # generate wav2vec mask and encoder mask
        downsampled_bucket_length = get_feat_extract_output_lengths(
            padding_length, kernel_size, stride
        )
        xs_len_ds = np.array(
            get_feat_extract_output_lengths(xs_lengths, kernel_size, stride)
        )
        padding_mask = make_pad_mask([xs_len_ds], max_len=downsampled_bucket_length)
        padding_mask = np.expand_dims(~padding_mask, 1)
        xs_masks = padding_mask.astype(np.float32)
        xs_lengths = np.array([xs_lengths])

        return uutid, xs_pad, xs_masks, tokens, xs_lengths

    ds = ds.map(
        operations=data_preprocess_e2e,
        input_columns=["uutid", "wav_path", "tokens"],
        output_columns=["uutid", "xs_pad", "xs_masks", "tokens", "xs_lengths"],
        column_order=["uutid", "xs_pad", "xs_masks", "tokens", "xs_lengths"],
        num_parallel_workers=num_workers,
    )
    return ds


def create_asr_predict_dataset(data_file, dataset_conf, collate_conf, num_workers=1):
    """Create ASR predictiong dataset.

    Args:
        data_file (str): input data file.
        extractor_conf (dict): configuration for feature extraction.
        dataset_conf (dict): configurations for the dataset.
        num_workers (int): number of process workers.

    Returns:
        A iterable data generator
    """
    dataset = AsrPredictDataset(
        data_file,
        dataset_conf["max_length"],
        dataset_conf["min_length"],
        dataset_conf["token_max_length"],
        dataset_conf["token_min_length"],
        100,
    )

    ds = de.GeneratorDataset(
        dataset,
        ["uutid", "wav_path", "length", "tokens"],
        max_rowsize=12,
        shuffle=False,
    )

    frame_bucket_limit = [int(i) for i in dataset_conf["frame_bucket_limit"].split(",")]

    def data_preprocess_asr(uutid, wav_path, length, tokens):
        # load wav data
        waveform, sample_rate = mindaudio.read(wav_path.item(0))
        waveform = waveform * (1 << 15)

        xs = compute_fbank_feats(
            waveform,
            sample_rate,
            mel_bin=int(collate_conf.feature_extraction_conf["mel_bins"]),
            frame_len=int(collate_conf.feature_extraction_conf["frame_length"]),
            frame_shift=int(collate_conf.feature_extraction_conf["frame_shift"]),
        )

        # batch size equals to 1
        # pad wavs
        padding_length = get_padding_length(length, frame_bucket_limit)
        xs_pad = pad_sequence(
            [xs],
            batch_first=True,
            padding_value=0.0,
            padding_max_len=padding_length,
            atype=np.float32,
        )
        xs_lengths = np.array([x.shape[0] for x in [xs]], dtype=np.int32)
        # make xs_masks, (B, 1, T), audio == 1, padding == 0
        xs_masks = np.expand_dims(~make_pad_mask(xs_lengths, max_len=padding_length), 1)
        return uutid, xs_pad, xs_masks, tokens, xs_lengths

    ds = ds.map(
        operations=data_preprocess_asr,
        input_columns=["uutid", "wav_path", "length", "tokens"],
        output_columns=["uutid", "xs_pad", "xs_masks", "tokens", "xs_lengths"],
        column_order=["uutid", "xs_pad", "xs_masks", "tokens", "xs_lengths"],
        num_parallel_workers=num_workers,
    )

    return ds
