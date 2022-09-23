import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype

class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """

    def __init__(self, network):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = ops.CTCLoss(ctc_merge_repeated=True)
        self.network = network
        self.ReduceMean_false = ops.ReduceMean(keep_dims=False)
        self.squeeze_op = ops.Squeeze(0)
        self.cast_op = ops.Cast()

    def construct(self, inputs, input_length, target_indices, label_values):
        predict, output_length = self.network(inputs, input_length)
        loss = self.loss(predict, target_indices, label_values, self.cast_op(output_length, mstype.int32))
        return self.ReduceMean_false(loss[0])


class PredictWithSoftmax(nn.Cell):
    """
    PredictWithSoftmax
    """

    def __init__(self, network):
        super(PredictWithSoftmax, self).__init__(auto_prefix=False)
        self.network = network
        self.inference_softmax = ops.Softmax(axis=-1)
        self.transpose_op = ops.Transpose()
        self.cast_op = ops.Cast()

    def construct(self, inputs, input_length):
        x, output_sizes = self.network(inputs, self.cast_op(input_length, mstype.int32))
        x = self.inference_softmax(x)
        x = self.transpose_op(x, (1, 0, 2))
        return x, output_sizes