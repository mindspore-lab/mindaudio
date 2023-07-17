""" Training Wrapper """
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context


class TrainingWrapper(nn.Cell):
    """
    Wraps the network with an optimizer
     Args:
       network (Cell): The training network.
       optimizer (Cell): Optimizer for updating the weights.
       sens (Number): The scaling number to be filled as the input of backpropagation. Default value is 1.0.
       use_global_norm: Whether to use global grad clip norm
       clip_global_norm_value : The value of global clip norm
    """

    def __init__(
        self,
        network,
        optimizer,
        sens=1.0,
        use_global_norm=True,
        clip_global_norm_value=5.0,
    ):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = float(sens)
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.clip_global_norm_value = clip_global_norm_value
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
            ParallelMode.DATA_PARALLEL,
            ParallelMode.HYBRID_PARALLEL,
        ]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(
                optimizer.parameters, mean, degree
            )

    def construct(self, *inputs):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = ops.clip_by_global_norm(
                grads, clip_norm=self.clip_global_norm_value
            )
        self.optimizer(grads)
        return loss
