# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ASR Predict Data Generator."""

import mindspore.dataset.engine as de
import numpy as np

from mindaudio.data.io import read
from flyspeech.adapter.log import get_logger
from flyspeech.dataset.bucket_dataset import get_padding_length, parse_file
from flyspeech.dataset.feature import compute_fbank_feats
from flyspeech.utils.common import get_feat_extract_output_lengths, pad_sequence
from flyspeech.utils.mask import make_pad_mask

logger = get_logger()


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

    def __init__(self,
                 data_file,
                 max_length=10240,
                 min_length=0,
                 token_max_length=200,
                 token_min_length=1,
                 frame_factor=100):
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
                logger.warning('Utts %s has %d frames, out of frame limit %d ~ %d, remove it.', data[i][0], length,
                               min_length, max_length)
                continue
            elif token_length > token_max_length or token_length < token_min_length:
                logger.warning('Utts %s has %d tokens, out of token limit %d ~ %d, remove it.', data[i][0],
                               token_length, token_min_length, token_max_length)
                continue
            else:
                num_sample += 1
                self.batches.append((uttid, wav_path, length, tokens))

        logger.info('Total utts: %d, remove too long/short utts: %d.', num_sample, len(data) - num_sample)

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
    with open(dict_file, 'r') as fin:
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
        dataset_conf['max_length'],
        dataset_conf['min_length'],
        dataset_conf['token_max_length'],
        dataset_conf['token_min_length'],
        16000,
    )

    ds = de.GeneratorDataset(
        dataset,
        ['uutid', 'wav_path', 'length', 'tokens'],
        max_rowsize=12,
        shuffle=False,
    )

    kernel_size = [int(i) for i in extractor_conf['kernel_size_list'].split(',')]
    stride = [int(i) for i in extractor_conf['stride_list'].split(',')]
    frame_bucket_limit = [int(i) for i in dataset_conf['frame_bucket_limit'].split(',')]

    def data_preprocess_e2e(uutid, wav_path, tokens):
        # load wav data
        waveform, _ = read(wav_path.item(0))
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
        downsampled_bucket_length = get_feat_extract_output_lengths(padding_length, kernel_size, stride)
        xs_len_ds = np.array(get_feat_extract_output_lengths(xs_lengths, kernel_size, stride))
        padding_mask = make_pad_mask([xs_len_ds], max_len=downsampled_bucket_length)
        padding_mask = np.expand_dims(~padding_mask, 1)
        xs_masks = padding_mask.astype(np.float32)
        xs_lengths = np.array([xs_lengths])

        return uutid, xs_pad, xs_masks, tokens, xs_lengths

    ds = ds.map(
        operations=data_preprocess_e2e,
        input_columns=['uutid', 'wav_path', 'tokens'],
        output_columns=['uutid', 'xs_pad', 'xs_masks', 'tokens', 'xs_lengths'],
        column_order=['uutid', 'xs_pad', 'xs_masks', 'tokens', 'xs_lengths'],
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
        dataset_conf['max_length'],
        dataset_conf['min_length'],
        dataset_conf['token_max_length'],
        dataset_conf['token_min_length'],
        100,
    )

    ds = de.GeneratorDataset(
        dataset,
        ['uutid', 'wav_path', 'length', 'tokens'],
        max_rowsize=12,
        shuffle=False,
    )

    frame_bucket_limit = [int(i) for i in dataset_conf['frame_bucket_limit'].split(',')]

    def data_preprocess_asr(uutid, wav_path, length, tokens):
        # load wav data
        waveform, sample_rate = read(wav_path.item(0))
        waveform = waveform * (1 << 15)

        xs = compute_fbank_feats(
            waveform,
            sample_rate,
            mel_bin=int(collate_conf.feature_extraction_conf['mel_bins']),
            frame_len=int(collate_conf.feature_extraction_conf['frame_length']),
            frame_shift=int(collate_conf.feature_extraction_conf['frame_shift']),
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
        input_columns=['uutid', 'wav_path', 'length', 'tokens'],
        output_columns=['uutid', 'xs_pad', 'xs_masks', 'tokens', 'xs_lengths'],
        column_order=['uutid', 'xs_pad', 'xs_masks', 'tokens', 'xs_lengths'],
        num_parallel_workers=num_workers,
    )

    return ds
