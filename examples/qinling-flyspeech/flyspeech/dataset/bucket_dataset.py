# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Chao Yang)
# Copyright (c) 2021 Jinsong Pan
# 2022.07 - Sample each batch by a pre-defined bucket
#           Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base bucket dataset definition."""

import codecs
import math
import os
from multiprocessing import Pool

from flyspeech.adapter.log import get_logger

logger = get_logger()


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
