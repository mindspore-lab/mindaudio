import numpy as np
import mindspore as ms


class DistributedSampler:
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
            indices = np.random.permutation(self.dataset_len)
        else:
            indices = np.arange(self.dataset_len)
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_len


def create_base_dataset(
    ds,
    rank: int = 0,
    group_size: int = 1,
):
    input_columns = ["audio", "text"]
    sampler = DistributedSampler(ds, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(
        ds,
        column_names=input_columns,
        sampler=sampler
    )
    return ds
