"""learning rate generator"""
import numpy as np


def get_lr(lr_init, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    half_epoch = total_epochs // 2
    for i in range(total_epochs):
        for _ in range(steps_per_epoch):
            if i < half_epoch:
                lr_each_step.append(lr_init)
            else:
                lr_each_step.append(lr_init / (1.1 ** (i - half_epoch)))
    learning_rate = np.array(lr_each_step).astype(np.float32)
    return learning_rate
