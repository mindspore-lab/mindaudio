"""CTC layer."""

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.operations as ops
import numpy as np


class CTC(nn.Cell):
    """CTC module."""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        compute_type=mstype.float32,
    ):
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
        self.ctc_loss = ops.CTCLossV2(blank=0, reduction="none", zero_infinity=True)
        self.log_softmax = nn.LogSoftmax(axis=2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.cast = ops.Cast()

    def construct(
        self,
        hs_pad: mindspore.Tensor,
        hlens: mindspore.Tensor,
        ys_pad: mindspore.Tensor,
        ys_lengths: mindspore.Tensor,
    ) -> mindspore.Tensor:
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
        ys_hat = ys_hat.transpose(1, 0, 2)
        ys_hat = self.log_softmax(ys_hat)
        loss, _ = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lengths)

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
