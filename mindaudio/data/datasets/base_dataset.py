""" Base dataset definition."""

import os
import mindspore as ms
from mindaudio.data.samplers import DistributedSampler


TAB = '\t'


class AudioTextBaseDataset():
    '''
    Each item of this dataset consists of a tuple (audio_path, trans_path):
        audio_path: str, path to audio file
        trans_path: str, path to transcript file
    '''
    def __init__(self, manifest_path=''):
        self.bins = []
        self.manifest_path = manifest_path
        # self.maybe_download()
        self.maybe_create_manifest()
        self.collect_data()

    def maybe_create_manifest(self):
        raise NotImplementedError

    def collect_data(self):
        with open(self.manifest_path) as file:
            for line in file.readlines():
                audio_path, trans_path = line.strip().split(TAB)
                self.bins.append([audio_path, trans_path])

    def maybe_download(self):
        if os.path.isdir(self.path):
            print('[download] found at', self.path)
            return
        try:
            print('[download] try downloading data...')
            os.makedirs(self.path, exist_ok=True)
            pass
        except:
            raise ValueError('no data downloaded, abort')

    def __getitem__(self, index):
        audio_path, trans_path = self.bins[index]
        return audio_path, trans_path

    def __len__(self):
        return len(self.bins)


def create_asr_dataset(
    ds: AudioTextBaseDataset,
    rank: int = 0,
    group_size: int = 1,
):
    '''
    Args:
        rank (int): current rank of the total world size, for multi-GPUs distributed training.
        group_size (int): number of total world size, for multi-GPUs distributed training.
    '''

    input_columns = ["audio", "text"]
    sampler = DistributedSampler(ds, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(
        ds,
        column_names=input_columns,
        sampler=sampler
    )

    return ds
