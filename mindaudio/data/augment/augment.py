from mindspore.dataset.audio import Resample
import numpy as np


def speed_perturb(waveform, sample_rate, speeds):
    speed = np.random.choice(speeds)
    if speed != 1.0:
        waveform = Resample(sample_rate, sample_rate * speed)(waveform)
    return waveform


def spec_aug(x, spec_aug_conf):
    """Do specaugment. Inplace operation

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
    num_t_mask = spec_aug_conf["num_t_mask"]
    num_f_mask = spec_aug_conf["num_f_mask"]
    prop_mask_t = spec_aug_conf["prop_mask_t"]
    prop_mask_f = spec_aug_conf["prop_mask_f"]
    max_frames = x.shape[0]
    max_freq = x.shape[1]
    # time mask
    for _ in range(num_t_mask):
        start = np.random.randint(0, max_frames - 1)
        length = np.random.randint(1, int(prop_mask_t * max_frames))
        end = min(max_frames, start + length)
        if np.random.randint(1, 100) > 20:
            x[start:end, :] = 0
    # freq mask
    for _ in range(num_f_mask):
        start = np.random.randint(0, max_freq - 1)
        length = np.random.randint(1, int(prop_mask_f * max_frames))
        end = min(max_freq, start + length)
        if np.random.randint(1, 100) > 20:
            x[:, start:end] = 0
    return x