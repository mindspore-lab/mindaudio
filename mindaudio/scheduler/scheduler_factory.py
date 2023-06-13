"""learning rate generator"""
import mindspore
import mindspore.ops as ops
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.nn.learning_rate_schedule import (
    CosineDecayLR,
    LearningRateSchedule,
    PolynomialDecayLR,
    WarmUpLR,
)


class ASRWarmupLR(LearningRateSchedule):
    """The WarmupLR scheduler

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

    def __init__(
        self,
        learninig_rate: float = 0.001,
        warmup_steps: int = 25000,
        start_steps: int = 0,
    ):
        super(ASRWarmupLR, self).__init__()
        self.learninig_rate = learninig_rate
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))
        self.min = ops.Minimum()
        self.scalar_summary = ops.ScalarSummary()
        self.start_steps = start_steps

    def construct(self, global_step):
        """construct asrwarmup scheduler"""
        step_num = global_step + self.start_steps
        warmup_percent = self.warmup_steps**0.5 * self.min(
            step_num**-0.5, step_num * self.warmup_steps**-1.5
        )
        current_lr = self.learninig_rate * warmup_percent

        return current_lr


class ASRLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """

    def __init__(
        self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power
    ):
        super(ASRLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(
            learning_rate, end_learning_rate, decay_steps, power
        )
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()
        self.scalar_summary = ops.ScalarSummary()

    def construct(self, global_step):
        """construct ASRLearningRate scheduler"""
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(
                self.greater(self.warmup_steps, global_step), mindspore.float32
            )
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        self.scalar_summary("learning_rate", lr)
        return lr


class CosineLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps):
        super(CosineLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = CosineDecayLR(end_learning_rate, learning_rate, decay_steps)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = ops.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = ops.Cast()
        self.scalar_summary = ops.ScalarSummary()

    def construct(self, global_step):
        """construct a CosineLearningRate scheduler"""
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(
                self.greater(self.warmup_steps, global_step), mindspore.float32
            )
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        self.scalar_summary("learning_rate", lr)
        return lr


def step_lr(lr_init, total_epochs, steps_per_epoch):
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
