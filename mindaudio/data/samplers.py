import numpy as np


class DistributedSampler:
    """Function to distribute and shuffle sample

    Args:
        dataset (BucketAsrDataset): init dataset instance.
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
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_len
