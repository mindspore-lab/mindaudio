import mindspore.ops as ops
from typing import List, Tuple

import mindspore
import numpy as np
import codecs
import math
import os
from multiprocessing import Pool
import multiprocessing as mp
import random

#import librosa
import mindspore.dataset.engine as de
import csv

from mindaudio.data.io import read
from mindaudio.data.processing import resample
from mindaudio.data.features import fbank

IGNORE_ID = -1

TILE_FUNC = ops.Tile()
CAT_FUNC = ops.Concat(axis=1)
MUL_FUNC = ops.Mul()
ADD_FUNC = ops.Add()
EQUAL_FUNC = ops.Equal()
ZEROSLIKE_FUNC = ops.ZerosLike()
CAST_FUNC = ops.Cast()
NEG_INF = mindspore.Tensor([-10000.0], dtype=mindspore.float32)


def subsequent_mask(size: int):
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    Args:
        size (int): size of mask

    Returns:
        np.ndarray: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    seq_range = np.arange(size)
    seq_range_expand = np.tile(seq_range, (size, 1))
    seq_length_expand = np.expand_dims(seq_range, -1)
    mask = seq_range_expand <= seq_length_expand
    return mask


def make_pad_mask(lengths: List[int], max_len: int = 0):
    """Make mask containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (List[int]): Batch of lengths (B,).
    Returns:
        np.ndarray: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = int(len(lengths))
    max_len = max_len if max_len > 0 else max(lengths)
    seq_range = np.expand_dims(np.arange(0, max_len), 0)
    seq_range_expand = np.tile(seq_range, (batch_size, 1))
    seq_length_expand = np.expand_dims(lengths, -1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: List[int], max_len: int = 0):
    """Make mask containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (List[int]): Batch of lengths (B,).
    Returns:
        np.ndarray: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths, max_len)


def mask_finished_scores(score: mindspore.Tensor, end_flag: mindspore.Tensor):
    """If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (mindspore.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (mindspore.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        mindspore.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.shape[-1]
    zero_mask = ZEROSLIKE_FUNC(end_flag)
    if beam_size > 1:
        unfinished = CAT_FUNC((zero_mask, TILE_FUNC(end_flag, (1, beam_size - 1))))
        finished = CAT_FUNC((end_flag, TILE_FUNC(zero_mask, (1, beam_size - 1))))
    else:
        unfinished = zero_mask
        finished = end_flag
    score = ADD_FUNC(score, MUL_FUNC(unfinished, NEG_INF))
    score = MUL_FUNC(score, (1 - finished))

    return score


def mask_finished_preds(pred: mindspore.Tensor, end_flag: mindspore.Tensor, eos: int) -> mindspore.Tensor:
    """If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (mindspore.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (mindspore.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        mindspore.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.shape[-1]
    finished = CAST_FUNC(TILE_FUNC(end_flag, (1, beam_size)), mindspore.int32)
    pred = pred * (1 - finished) + eos * finished
    return pred


def compute_mask_indices(shape, mask_prob, mask_length) -> np.ndarray:
    """compute mask indices."""
    b, t = shape
    mask = np.full((b, t), False)
    n_mask = int(mask_prob * t / float(mask_length) + 0.35)
    for i in range(b):
        ti = t
        span = ti // n_mask
        for j in range(n_mask):
            # non-overlaped masking
            start = j * span + np.random.randint(span - mask_length)
            mask[i][start:start + mask_length] = True
    return mask


def compute_mask_indices2(shape, padding_mask, mask_prob, mask_length) -> np.ndarray:
    """compute mask indices2."""
    b, t = shape
    mask = np.full((b, t), False)
    mask_valid = np.full((b, t), False)
    n_mask = int(mask_prob * t / float(mask_length) + 0.35)
    for i in range(b):
        real_wav_len = t - padding_mask[i].astype(int).sum().item()
        ti = t
        span = ti // n_mask
        for j in range(n_mask):
            # non-overlaped masking
            start = j * span + np.random.randint(span - mask_length)
            mask[i][start:start + mask_length] = True
        mask_valid[i][:real_wav_len] = True
    return mask, mask_valid


def subsequent_chunk_mask(size: int, chunk_size: int, num_left_chunks: int = -1):
    """Create mask for subsequent steps (size, size) with chunk size, this is
    for streaming encoder.

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks

    Returns:
        numpy.array: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = np.zeros((size, size), dtype=np.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs_len, masks, decoding_chunk_size,
                            static_chunk_size, num_decoding_left_chunks):
    """Apply optional mask for encoder.

    Args:
        xs_len (int): padded input, 1/4 ori data length
        mask (numpy.array): mask for xs, (B, 1, L)

        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks

    Returns:
        numpy.array: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    masks = masks.astype(np.bool)

    if static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs_len, static_chunk_size, num_left_chunks)  # (L, L)
        chunk_masks = np.expand_dims(chunk_masks, 0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


def pad_sequence(sequences: List[np.ndarray],
                 batch_first=True,
                 padding_value: int = 0,
                 padding_max_len: int = None,
                 atype=np.int32) -> np.ndarray:
    """[summary]

    Args:
        sequences (List[np.ndarray]): [description]
        batch_first (bool, optional): [description]. Defaults to True.
        padding_value (int, optional): [description]. Defaults to 0.
        padding_max_len (int, optional): [description]. Defaults to None.
        atype ([type], optional): [description]. Defaults to np.int32.

    Returns:
        np.ndarray: [description]
    """
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]

    if padding_max_len is not None:
        max_len = padding_max_len
    else:
        max_len = max([s.shape[0] for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_sequences = np.full(out_dims, fill_value=padding_value).astype(atype)

    for i, seq in enumerate(sequences):
        length = seq.shape[0] if seq.shape[0] <= max_len else max_len
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_sequences[i, :length, ...] = seq[:length]
        else:
            out_sequences[:length, i, ...] = seq[:length]

    return out_sequences


def add_sos_eos(ys: List[np.ndarray], sos: int = 0, eos: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Add <sos> and <eos> labels. For generating the decoder input and output.

    Args:
        ys (List[np.ndarray]): list of target sequences
        sos (int): index of <sos>
        eos (int): index of <eos>

    Returns:
        ys_in (List[np.ndarray])
        ys_out (List[np.ndarray])

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ys_pad
        [array([ 1,  2,  3,  4,  5]),
         array([ 4,  5,  6]),
         array([ 7,  8,  9])]
        >>> ys_in, ys_out = add_sos_eos(ys_pad, sos_id, eos_id)
        >>> ys_in
        [array([10,  1,  2,  3,  4,  5]),
         array([10,  4,  5,  6]),
         array([10,  7,  8,  9])]
        >>> ys_out
        [array([ 1,  2,  3,  4,  5, 11]),
         array([ 4,  5,  6, 11]),
         array([ 7,  8,  9, 11])]
    """
    ys_in = [np.concatenate(([sos], y), axis=0) for y in ys]
    ys_out = [np.concatenate((y, [eos]), axis=0) for y in ys]
    return ys_in, ys_out


class DistributedSampler:
    """Function to distribute and shuffle sample.

    Args:
        dataset (BucketDataset): init dataset instance.
        rank (int): current rank of the total world size, for multi-GPUs distributed training
        group_size (int): number of total world size, for multi-GPUs distributed training.
        shuffle (bool): whether to shuffle the dataset.
        seed (int): random seed.
    """

    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xFFFFFFFF
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len).tolist()
        else:
            indices = list(range(self.dataset_len))
        return iter(indices)

    def __len__(self):
        return self.dataset_len


def parallel_compute_fbank_feats(param):
    """compute fbank feats."""
    wav_path = param[0][1]
    use_speed_perturb = param[1]

    waveform, sample_rate = read(wav_path)
    #waveform = waveform * (1 << 15)

    if use_speed_perturb:
        waveform = speed_perturb(waveform, sample_rate=16000)

    return fbank(waveform, sample_rate)


def speed_perturb(waveform, sample_rate):
    """speed perturb."""
    speeds = [0.9, 1.0, 1.1]
    speed = random.choice(speeds)

    if speed != 1.0:
        waveform = resample(waveform, sample_rate * speed, sample_rate)
    return waveform


def parse_file(path):
    """parse file."""
    assert os.path.exists(path)
    num = 0
    results = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if num > 0:
                results += [row]
            num += 1
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

    def __init__(self,
                 data_file,
                 frame_bucket_limit='200,300',
                 batch_bucket_limit='220,200',
                 batch_factor=0.2,
                 frame_factor=100,
                 group_size=1):
        self.group_size = group_size
        self.frame_bucket_limit = [int(i) for i in frame_bucket_limit.split(',')]
        self.batch_bucket_limit = [int(int(i) * batch_factor * group_size) for i in batch_bucket_limit.split(',')]
        assert len(self.frame_bucket_limit) == len(self.batch_bucket_limit)
        self.bucket_select_dict = self.bucket_init(self.frame_bucket_limit)

        # load all samples
        data = parse_file(data_file)
        # sort all date according to their lengths
        # each item of data include [uttid, wav_path, duration, tokenid, output_dim, kmeans_id (optional)]
        self.data = sorted(data, key=lambda x: x[1])
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

    def __init__(self,
                 data_file,
                 max_length=10240,
                 min_length=0,
                 token_max_length=200,
                 token_min_length=1,
                 frame_bucket_limit='200,300',
                 batch_bucket_limit='220,200',
                 batch_factor=0.2,
                 frame_factor=100,
                 group_size=1):
        super().__init__(data_file,
                         frame_bucket_limit=frame_bucket_limit,
                         batch_bucket_limit=batch_bucket_limit,
                         batch_factor=batch_factor,
                         frame_factor=frame_factor,
                         group_size=group_size)
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
            length = int(int(self.data[i][2])/80)
            token_length = len(self.data[i][3].split())
            self.output_dim = len(self.data[i][3].split())
            # remove too lang or too short utt for both input and output
            if length > max_length or length < min_length:
                continue
            elif token_length > token_max_length or token_length < token_min_length:
                continue
            else:
                num_sample += 1
                bucket_idx = self.bucket_select_dict[length]
                caches[bucket_idx][0].append((self.data[i][0], self.data[i][1], self.data[i][3]))
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
                self.batches.append((data_expand[:self.batch_bucket_limit[key]], value[2]))
        del caches

        self.sos = self.data[i][5]
        self.eos = self.data[i][6]

    def __getitem__(self, index):
        data, max_src_len = self.batches[index][0], self.batches[index][1]
        return data, self.sos, self.eos, max_src_len, self.token_max_length

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


        decoding_chunk_size (int): number of chunk use for inference, for streaming ASR.
        static_chunk_size (int): chunk size for streaming ASR.
        num_decoding_left_chunks (int):  number of left chunk use for inference, for streaming ASR.
    """

    def __init__(self,
                 rank,
                 group_size,
                 feature_extraction_conf=None,
                 feature_dither=0.0,
                 use_speed_perturb=False,
                 use_spec_aug=False,
                 spec_aug_conf=None,


                 decoding_chunk_size=0,
                 static_chunk_size=0,
                 num_decoding_left_chunks=-1):
        self.feature_extraction_conf = feature_extraction_conf
        self.feature_dither = feature_dither
        self.use_speed_perturb = use_speed_perturb
        self.use_spec_aug = use_spec_aug
        self.spec_aug_conf = spec_aug_conf
        self.rank = rank
        self.group_size = group_size
        self.pool = mp.Pool(8)


        self.decoding_chunk_size = decoding_chunk_size
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def extract_feature(self, batch, use_speed_perturb):
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

        mel_bin = 80
        frame_len = 10
        frame_shift = 25
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
        print(labels)
        labels = [np.fromiter(map(int, x), dtype=np.int32) for x in labels]
        sorted_labels = [labels[i] for i in order]

        return sorted_uttids, sorted_feats, sorted_labels


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
            batch[self.rank::self.group_size],
            self.use_speed_perturb,
        )
        if self.feature_dither != 0.0:
            raise NotImplementedError

        # if self.use_spec_aug:
        #     xs = self.spec_aug(xs, self.spec_aug_conf)

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
        ys_masks = np.expand_dims(~make_pad_mask(ys_lengths + 1, max_len=max_tgt_len + 1), 1)
        m = np.expand_dims(subsequent_mask(max_tgt_len + 1), 0)
        ys_sub_masks = (ys_masks & m).astype(np.float32)
        ys_masks = ys_masks.astype(np.float32)
        xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]
        xs_pad_downsample4_len = (xs_pad.shape[1] - 3) // 4
        xs_chunk_masks = add_optional_chunk_mask(
            xs_len=xs_pad_downsample4_len,
            masks=xs_masks,

            decoding_chunk_size=self.decoding_chunk_size,
            static_chunk_size=self.static_chunk_size,
            num_decoding_left_chunks=self.num_decoding_left_chunks,
        )

        return xs_pad, ys_pad, ys_in_pad, ys_out_pad, r_ys_in_pad, r_ys_out_pad, \
            xs_masks, ys_sub_masks, ys_masks, ys_lengths, xs_chunk_masks

def create_dataset(data_file, rank=0, group_size=1, number_workers=8):
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
    collate_func = CollateFunc(rank=rank, group_size=group_size)
    dataset = BucketASRDataset(
        data_file,
        max_length=3000,
        min_length=0,
        token_max_length=30,
        token_min_length=1,
        frame_bucket_limit='144, 204, 288, 400, 512, 600, 712, 800, 912, 1024, 1112, 1200, 1400, 1600, 2000, 3000',
        batch_bucket_limit='40, 80, 80, 72, 72, 56, 56, 56, 40, 40, 40, 40, 24, 8, 8, 8',
        batch_factor=1,
        frame_factor=100,
        group_size=group_size,
    )

    sampler = DistributedSampler(dataset, rank, group_size, shuffle=True)

    ds = de.GeneratorDataset(
        dataset,
        ['data', 'sos', 'eos', 'max_src_len', 'token_max_length'],
        sampler=sampler,
        num_parallel_workers=1,
        max_rowsize=24,
    )
    iterator = ds.create_dict_iterator()
    num = 0
    for item in iterator:
        if num < 2:
            print(item)
        num = num + 1

    output_dim = dataset.output_dim

    # pylint: disable=W0108
    map_func = lambda data, sos, eos, max_src_len, token_max_length: collate_func(data, sos, eos, max_src_len,
                                                                                  token_max_length)
    ds = ds.map(operations=map_func,
                input_columns=['data', 'sos', 'eos', 'max_src_len', 'token_max_length'],
                output_columns=[
                    'xs_pad', 'ys_pad', 'ys_in_pad', 'ys_out_pad', 'r_ys_in_pad', 'r_ys_out_pad',
                    'xs_masks', 'ys_masks', 'ys_sub_masks',
                    'ys_lengths',
                    'xs_chunk_masks'
                ],
                column_order=[
                    'xs_pad', 'ys_pad', 'ys_in_pad', 'ys_out_pad', 'r_ys_in_pad', 'r_ys_out_pad',
                    'xs_masks', 'ys_masks', 'ys_sub_masks',
                    'ys_lengths',
                    'xs_chunk_masks'
                ],
                num_parallel_workers=number_workers,
                python_multiprocessing=False)

    return output_dim, ds

output_dim , dataset = create_dataset('/home/litingyu/data/test_csv/train.csv')
iterator = dataset.create_dict_iterator()
num = 0
for item in iterator:
    if num < 10:
        print(item)
    num = num+1