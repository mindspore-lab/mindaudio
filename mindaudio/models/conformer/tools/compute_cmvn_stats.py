# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""compute cmvn stats methods."""

import sys
import argparse
import json
import codecs
import yaml
import librosa
import numpy as np

import mindspore.dataset.engine as de

from src.dataset.sampler import DistributedSampler
from src.dataset.feature import compute_fbank_feats


class AudioDataset:
    """audio dataset definition.

    Args:
        data_file (string): audio data file path
    """
    def __init__(self, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split()
                self.items.append((arr[1],))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='',
                        help='training yaml conf')
    parser.add_argument('--in_scp', default=None, help='wav scp file')
    parser.add_argument('--out_cmvn',
                        default='global_cmvn',
                        help='global cmvn file')

    doc = "Print log after every log_interval audios are processed."
    parser.add_argument("--log_interval", type=int, default=1000, help=doc)
    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_dim = configs['collate_conf']['feature_extraction_conf']['mel_bins']

    resample_rate = 0
    if 'resample_conf' in configs['dataset_conf']:
        resample_rate = configs['dataset_conf']['resample_conf']['resample_rate']
        print('using resample and new sample rate is {}'.format(resample_rate))

    collate_conf = configs['collate_conf']
    dataset_conf = configs['dataset_conf']

    dataset = AudioDataset(args.in_scp)
    batch_size = 1
    sampler = DistributedSampler(dataset, 0, 1, shuffle=False)
    ds = de.GeneratorDataset(dataset,
                             ['wav_path'],
                             sampler=sampler,
                             num_parallel_workers=1,
                             max_rowsize=24)

    def data_preprocess_asr(wav_path):
        """data preprocess.

        This method is used to load audio files, extract fbank features, statistical means and variance information.

        Args:
            wav_path (string): audio file path

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): number, mean_stat, var_stat
        """
        mean_stat = np.zeros(feat_dim)
        var_stat = np.zeros(feat_dim)
        number = 0
        waveform, sample_rate = librosa.load(wav_path.item(0), sr=16000)
        waveform = waveform * (1 << 15)

        xs = compute_fbank_feats(
            waveform,
            sample_rate,
            mel_bin=int(collate_conf["feature_extraction_conf"]["mel_bins"]),
            frame_len=int(
                collate_conf["feature_extraction_conf"]["frame_length"]),
            frame_shift=int(
                collate_conf["feature_extraction_conf"]["frame_shift"]),
        )

        mean_stat += np.sum(xs, axis=0)
        var_stat += np.sum(np.square(xs), axis=0)
        number += xs.shape[0]
        number = np.array(number)
        return number, mean_stat, var_stat

    ds = ds.map(
        operations=data_preprocess_asr,
        input_columns=["wav_path"],
        output_columns=["number", "mean_stat", "var_stat"],
        column_order=["number", "mean_stat", "var_stat"],
        num_parallel_workers=1,
    )

    all_number = 0
    all_mean_stat = np.zeros(feat_dim)
    all_var_stat = np.zeros(feat_dim)
    wav_number = 0
    for data in ds.create_dict_iterator(output_numpy=True):
        all_mean_stat += data['mean_stat']
        all_var_stat += data['var_stat']
        all_number += data['number']
        wav_number += batch_size

        if wav_number % args.log_interval == 0:
            print(f'processed {wav_number} wavs, {all_number} frames',
                  file=sys.stderr,
                  flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number.tolist()
    }

    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))
