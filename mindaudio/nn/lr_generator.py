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


def get_tacotron2_lr(
    init_lr,
    total_epoch,
    step_per_epoch,
    anneal_step=250
):
    ''' warmup lr schedule'''
    total_step = total_epoch * step_per_epoch
    lr_step = []

    for step in range(total_step):
        lambda_lr = anneal_step**0.5 * \
            min((step + 1) * anneal_step**-1.5, (step + 1)**-0.5)
        lr_step.append(init_lr * lambda_lr)
    learning_rate = np.array(lr_step).astype(np.float32)
    return learning_rate
