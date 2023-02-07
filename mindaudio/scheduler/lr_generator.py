"""learning rate generator"""
import numpy as np
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import LearningRateSchedule


class ASRWarmupLR(LearningRateSchedule):
    """The WarmupLR scheduler.

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.
    """

    def __init__(self, learninig_rate: float = 0.001, warmup_steps: int = 25000, start_steps: int = 0):
        super(ASRWarmupLR, self).__init__()
        self.learninig_rate = learninig_rate
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.min = ops.Minimum()
        self.scalar_summary = ops.ScalarSummary()
        self.start_steps = start_steps

    def construct(self, global_step):
        """construct asrwarmup scheduler."""
        step_num = global_step + self.start_steps
        warmup_percent = self.warmup_steps**0.5 * self.min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
        current_lr = self.learninig_rate * warmup_percent
        return current_lr
    

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
