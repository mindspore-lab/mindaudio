"""CTC layer."""

import numpy as np
import mindspore
import mindspore.common.dtype as mstype
import mindspore.ops.operations as ops
import mindspore.nn as nn  # pylint: disable=C0412


class CTC(nn.Cell):
    """CTC module."""

    def __init__(self, odim: int, encoder_output_size: int, dropout_rate: float = 0.0, compute_type=mstype.float32):
        """Construct CTC module.

        Args:
            odim (int): dimension of outputs.
            encoder_output_size (int): number of encoder projection units.
            dropout_rate (float): dropout rate (0.0 ~ 1.0).
            compute_type (dtype): whether to do mix-precision training, mstype.float32
                                  or mstype.float16.
        """
        super().__init__()
        eprojs = encoder_output_size
        self.ctc_lo = nn.Dense(eprojs, odim).to_float(compute_type)
        self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)
        self.log_softmax = nn.LogSoftmax(axis=1)
        self.dropout = nn.Dropout(1.0 - dropout_rate)
        self.cast = ops.Cast()
        self.count = 0

    def construct(self, hs_pad: mindspore.Tensor, hlens: mindspore.Tensor, ys_pad: mindspore.Tensor,
                  ys_pad_indices: mindspore.Tensor) -> mindspore.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(self.dropout(hs_pad))
        ys_hat = self.cast(ys_hat, mstype.float32)
        # ys_hat: (B, L, D) -> (L, B, D)
        B, L, D = ys_hat.shape
        ys_hat = ys_hat.transpose(1, 0, 2).reshape((-1, D))
        ys_hat = self.log_softmax(ys_hat).reshape((L, B, D))
        ys_pad = ys_pad.reshape(-1).astype(mindspore.int32)
        loss, _ = self.ctc_loss(ys_hat, ys_pad_indices, ys_pad, hlens)

        # Batch-size average
        loss = loss.sum()
        loss = loss / ys_hat.shape[1]

        return loss

    def compute_log_softmax_out(self, hs_pad: mindspore.Tensor) -> mindspore.Tensor:
        """log_softmax of frame activations.

        Args:
            hs_pad (mindspore.Tensor): (batch_size, seq_length, hidden_dim)
        Returns:
            mindspore.Tensor: log softmax output (batch_size, seq_length, vocab_size)
        """
        ys_hat = self.cast(self.ctc_lo(hs_pad), mstype.float32)
        return self.log_softmax(ys_hat)

    def argmax(self, hs_pad: mindspore.Tensor) -> mindspore.Tensor:
        """argmax of frame activations.

        Args:
            hs_pad (mindspore.Tensor): 3d tensor (B, Tmax, eprojs)
        Returns:
            mindspore.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return np.argmax(self.ctc_lo(hs_pad), axis=2)


class NetWithCTCLoss(nn.Cell):
    """
    NetWithCTCLoss definition
    """

    def __init__(self, network, ascend=False):
        super(NetWithCTCLoss, self).__init__(auto_prefix=False)
        if ascend:
            self.loss = ops.CTCLoss(
                ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=True
            )
        else:
            self.loss = ops.CTCLoss(ctc_merge_repeated=True)
        self.network = network
        self.ReduceMean_false = ops.ReduceMean(keep_dims=False)
        self.squeeze_op = ops.Squeeze(0)
        self.cast_op = ops.Cast()

    def construct(self, inputs, input_length, target_indices, label_values):
        predict, output_length = self.network(inputs, input_length)
        predict = self.cast_op(predict, mstype.float32)
        loss = self.loss(
            predict,
            target_indices,
            label_values,
            self.cast_op(output_length, mstype.int32),
        )
        return self.ReduceMean_false(loss[0])
