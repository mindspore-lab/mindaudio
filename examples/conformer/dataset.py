"""ASR Training Data Generator."""

import multiprocessing as mp
import random

import mindspore.dataset.engine as de
import numpy as np
from adapter.log import get_logger
from dataset.bucket_dataset import BucketASRDataset
from dataset.feature import compute_fbank_feats
from dataset.sampler import DistributedSampler
from mindaudio.utils.common import IGNORE_ID, add_sos_eos, pad_sequence
from mindaudio.utils.mask import add_optional_chunk_mask, make_pad_mask, subsequent_mask

import mindaudio

logger = get_logger()


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
