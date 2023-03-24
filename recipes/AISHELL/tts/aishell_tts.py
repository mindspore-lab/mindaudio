import mindspore as ms

from mindaudio.utils.distributed import DistributedSampler


def create_aishell_tts_dataset(
    ds,
    rank: int = 0,
    group_size: int = 1,
):
    input_columns = ["audio", "text"]
    sampler = DistributedSampler(ds, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(ds, column_names=input_columns, sampler=sampler)
    return ds
