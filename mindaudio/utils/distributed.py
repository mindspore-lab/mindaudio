import numpy as np


class DistributedSampler:
    """
    For mindspore.dataset.GeneratorDataset
    """

    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0, group=True):
        self.group = group
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xFFFFFFFF
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len)
        else:
            indices = np.arange(self.dataset_len)
        if self.group:
            indices = indices[self.rank :: self.group_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_len
