""" Loss """
from itertools import permutations

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor

EPS = 1e-8


class Separation_Loss(nn.Cell):
    def __init__(self):
        super(Separation_Loss, self).__init__()
        self.mean = ops.ReduceMean()
        self.cast = ops.Cast()
        self.sum = ops.ReduceSum(keep_dims=True)
        self._sum = ops.ReduceSum(keep_dims=False)
        self.expand_dims = ops.ExpandDims()
        self.log = ops.Log()
        # self.scatter = ops.ScatterNd()
        self.matmul = ops.MatMul()
        self.transpose = ops.Transpose()
        self.Argmax = ops.Argmax(axis=1, output_type=mindspore.int32)
        self.argmax = ops.ArgMaxWithValue(axis=1, keep_dims=True)
        self.ones = ops.Ones()
        self.zeros_like = ops.ZerosLike()
        self.Print = ops.Print()
        # self.source_lengths_with_list = get_input_with_list(dir)
        self.log10 = Tensor(np.array([10.0]), mindspore.float32)
        self.perms = Tensor(list(permutations(range(2))), dtype=mindspore.int32)
        self.perms_ones_hot = Tensor(
            np.array([[1, 0], [0, 1], [0, 1], [1, 0]]), mindspore.float32
        )

    def construct(self, source, estimate_source, source_lengths):
        return self.cal_loss(source, estimate_source, source_lengths)

    def cal_loss(self, source, estimate_source, source_lengths):
        """
        Args:
            source: [B, C, K, L]
            estimate_source: [B, C, K, L]
            source_lengths: [B]
        """
        max_snr, perms, max_snr_idx = self.cal_si_snr_with_pit(
            source, estimate_source, source_lengths
        )
        loss_final = 0 - self.mean(max_snr)
        reorder_estimate_source = self.reorder_source(
            estimate_source, perms, max_snr_idx
        )
        return loss_final, max_snr, estimate_source, reorder_estimate_source

    def cal_si_snr_with_pit(self, source, estimate_source, source_lengths):
        """
        Calculate SI-SNR with PIT training.
        Args:
            source: [B, C, K, L]
            estimate_source: [B, C, K, L]
            source_lengths: [B], each item is between [0, K]
        """
        B, C, _, L = source.shape
        num_samples = (L * source_lengths).view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        mean_target = self.sum(source, [2, 3]) / num_samples
        mean_estimate = self.sum(estimate_source, [2, 3]) / num_samples
        zero_mean_target = source - mean_target
        zero_mean_estimate = estimate_source - mean_estimate

        # SI-SNR with PIT
        # flat K, L to T (T = K * L)
        flat_target = zero_mean_target.view(B, C, -1)  # [B, C, T]
        flat_estimate = zero_mean_estimate.view(B, C, -1)  # [B, C, T]
        # reshape to use broadcast
        s_target = self.expand_dims(flat_target, 1)  # [B, 1, C, T]
        s_estimate = self.expand_dims(flat_estimate, 2)  # [B, C, 1, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = self.sum(s_estimate * s_target, 3)  # [B, C, C, 1]
        s_target_energy = self.sum(s_target**2, 3) + EPS  # [B, 1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = self._sum(pair_wise_proj**2, 3) / (
            self._sum(e_noise**2, 3) + EPS
        )
        pair_wise_si_snr = (
            10 * self.log(pair_wise_si_snr + EPS) / self.log(self.log10)
        )  # [B, C, C]

        # Get max_snr of each utterance
        # permutations, [C!, C]
        perms = self.perms
        perms_one_hot = self.perms_ones_hot
        snr_set = self.matmul(pair_wise_si_snr.view(B, -1), perms_one_hot)
        max_snr_idx = self.Argmax(snr_set)
        _, max_snr = self.argmax(snr_set)
        max_snr /= C
        return max_snr, perms, max_snr_idx

    def get_mask(self, source):
        """Calculate mask"""
        B, _, K, _ = source.shape
        mask = self.ones((B, 1, K, 1), mindspore.float32)
        for i in range(B):
            mask[i, :, 3320:, :] = 0
        return mask

    def reorder_source(self, source, perms, max_snr_idx):
        """
        Args:
            source: [B, C, K, L]
            perms: [C!, C], permutations
            max_snr_idx: [B], each item is between [0, C!)
        Returns:
            reorder_source: [B, C, K, L]
        """
        B, C, _, _ = source.shape
        # [B, C], permutation whose SI-SNR is max of each utterance
        # for each utterance, reorder estimate source according this permutation
        max_snr_perm = perms[max_snr_idx, :]
        reorder_source = self.zeros_like(source)
        for b in range(B):
            for c in range(C):
                if max_snr_perm[b][c] == 1:
                    reorder_source[b, c] = source[b, 1]
                else:
                    reorder_source[b, c] = source[b, 0]
        return reorder_source


class NetWithLoss(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        net (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, net, loss_fn):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self._net = net
        self._loss = loss_fn
        self.cast = ops.Cast()

    def construct(self, padded_mixture, mixture_lengths, padded_source):
        estimate_source = self._net(padded_mixture)
        loss, _, estimate_source, _ = self._loss(
            padded_source, estimate_source, mixture_lengths
        )
        return loss
