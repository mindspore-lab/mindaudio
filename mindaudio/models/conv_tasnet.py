import argparse
import math

import mindspore
import mindspore.common.initializer
import numpy as np
from mindspore import Tensor, context, nn, ops

EPS = 1e-8


class ConvTasNet(nn.Cell):
    def __init__(
        self,
        N,
        L,
        B,
        H,
        P,
        X,
        R,
        C,
        norm_type="gLN",
        causal=False,
        mask_nonlinear="relu",
    ):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(ConvTasNet, self).__init__()
        # Hyper-parameter
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.C = C
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        # Components
        self.encoder = Encoder(L, N)
        self.separator = TemporalConvNet(
            N, B, H, P, X, R, C, norm_type, causal, mask_nonlinear
        )
        self.decoder = Decoder(N, L)
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (0, 10)), mode="CONSTANT")
        # init
        for p in self.get_parameters():
            if p.ndim > 1:
                mindspore.common.initializer.HeNormal(p)

    def construct(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)
        est_source = self.pad(est_source)
        return est_source


class Encoder(nn.Cell):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer."""

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L = L
        self.N = N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(
            1,
            N,
            kernel_size=L,
            stride=L // 2,
            has_bias=False,
            pad_mode="pad",
            weight_init="HeUniform",
        )
        self.relu = nn.ReLU()

    def construct(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples， N是通道数
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = ops.expand_dims(mixture, 1)  # [M, 1, T]
        mixture_w = self.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


def big_matrix():
    alpha = np.zeros((6398, 3199), np.float16)
    for i in range(3199):
        alpha[2 * i, i] = 1
        alpha[2 * i + 1, i] = 1
    beta = Tensor.from_numpy(alpha)
    return beta


class Decoder(nn.Cell):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N = N
        self.L = L
        # Components
        self.basis_signals = nn.Dense(N, L, has_bias=False)
        self.big_matrix = big_matrix()

    def construct(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]   K = (T-L)/(L/2)+1 = 2T/L-1
        Returns:
            est_source: [M, C, T]   #输出的【batch size，说话人数，T is #samples】
        """
        # D = W * M
        source_w = ops.expand_dims(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = ops.transpose(source_w, (0, 1, 3, 2))
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, L]
        est_source = self.overlap_and_add(est_source, self.L // 2)  # M x C x T
        return est_source

    def overlap_and_add(self, signal, frame_step):
        """Reconstructs a signal from a framed representation.

        Adds potentially overlapping frames of a signal with shape
        `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
        The resulting tensor has shape `[..., output_size]` where

            output_size = (frames - 1) * frame_step + frame_length

        Args:
            signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
            frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

        Returns:
            output_size = (frames - 1) * frame_step + frame_length

        """

        outer_dimensions = signal.shape[:-2]
        frames, frame_length = signal.shape[-2:]

        subframe_length = math.gcd(
            frame_length, frame_step
        )  # gcd=Greatest Common Divisor
        subframe_step = mindspore.Tensor(frame_step // subframe_length)
        output_size = frame_step * (frames - 1) + frame_length
        output_subframes = output_size // subframe_length

        subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
        frame = mindspore.numpy.arange(0, output_subframes)
        frame = ops.Concat(-1)(
            (
                ops.expand_dims(frame[0:-1:subframe_step], 1),
                ops.expand_dims(frame[1::subframe_step], 1),
            )
        )
        frame = frame.view(-1)
        subframe_signal_ = subframe_signal.transpose((0, 1, 3, 2))

        result_ = ops.matmul(subframe_signal_, self.big_matrix)
        result = result_.transpose(((0, 1, 3, 2)))
        result = result.view(*outer_dimensions, -1)
        return result


class TemporalConvNet(nn.Cell):
    def __init__(
        self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear="relu"
    ):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        self.softmax = nn.Softmax(axis=1)
        self.relu = nn.ReLU()
        # Components
        # [M, N, K] -> [M, N, K]
        self.layer_norm = GlobalLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1 = nn.Conv1d(
            N, B, kernel_size=1, stride=1, has_bias=False, weight_init="HeUniform"
        )
        # [M, B, K] -> [M, B, K]
        repeats = []
        while R > 0:
            R -= 1
            blocks = []
            for gamma in range(X):
                dilation = 2**gamma
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.SequentialCell(*blocks)]
        self.temporal_conv_net = nn.SequentialCell(*repeats)
        # [M, B, K] -> [M, C*N, K]
        self.mask_conv1x1 = nn.Conv1d(
            B, C * N, 1, has_bias=False, weight_init="HeUniform"
        )
        # Put together
        self.network = nn.SequentialCell(
            self.layer_norm,
            self.bottleneck_conv1x1,
            self.temporal_conv_net,
            self.mask_conv1x1,
        )

    def construct(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.shape
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        score = score.view(M, self.C, N, K)  # [M, C*N, K] -> [M, C, N, K]
        est_mask = self.relu(score)
        return est_mask


class TemporalBlock(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.norm_type = norm_type
        self.padding = padding
        self.conv1x1 = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            pad_mode="valid",
            has_bias=False,
            weight_init="HeUniform",
        )
        self.prelu = nn.PReLU()
        self.causal = causal
        self.norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        self.dsconv = DepthwiseSeparableConv(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.norm_type,
            self.causal,
        )
        # Put together
        self.net = nn.SequentialCell(self.conv1x1, self.prelu, self.norm, self.dsconv)

    def construct(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        x = self.conv1x1(x)
        x = self.prelu(x)
        x = self.norm(x)
        x = self.dsconv(x)
        t = x + residual
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return t  # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Cell):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.depthwise_conv = nn.Conv1d(
            self.in_channels,
            self.in_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            pad_mode="pad",
            dilation=self.dilation,
            group=self.in_channels,
            has_bias=False,
            weight_init="HeUniform",
        )
        self.prelu = nn.PReLU()
        self.norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        self.pointwise_conv = nn.Conv1d(
            in_channels, out_channels, 1, has_bias=False, weight_init="HeUniform"
        )
        # Put together

    def construct(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        x = self.depthwise_conv(x)
        x = self.prelu(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x


class Chomp1d(nn.Cell):
    """To ensure the output length is the same as the input."""

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def construct(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, : -self.chomp_size]


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
    C is channel size and K is sequence length.
    """
    return GlobalLayerNorm(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Cell):
    """Channel-wise Layer Normalization (cLN)"""

    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma1 = np.ones((1, channel_size, 1)).astype(np.float32)
        self.gamma2 = Tensor.from_numpy(self.gamma1)
        self.beta1 = np.zeros((1, channel_size, 1)).astype(np.float32)
        self.beta2 = Tensor.from_numpy(self.beta1)
        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = self.mean(y, 1)  # [M, 1, K]
        var = y.var(axis=1, keepdims=True, ddof=0)  # [M, 1, K]
        cLN_y = self.gamma2 * (y - mean) / ops.pow(var + EPS, 0.5) + self.beta2
        return cLN_y


class GlobalLayerNorm(nn.Cell):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma1 = np.ones((1, channel_size, 1)).astype(np.float32)
        self.gamma2 = Tensor.from_numpy(self.gamma1)
        self.beta1 = np.zeros((1, channel_size, 1)).astype(np.float32)
        self.beta2 = Tensor.from_numpy(self.beta1)
        self.mean = ops.ReduceMean(keep_dims=True)

    def construct(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        mean = self.mean(y, 1)
        mean = self.mean(mean, 2)
        var = ops.pow(y - mean, 2)
        var = self.mean(var, 1)
        var = self.mean(var, 2)
        gLN_y = self.gamma2 * (y - mean) / ops.pow(var + EPS, 0.5) + self.beta2
        return gLN_y


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
    parser = argparse.ArgumentParser(
        "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
        "with Permutation Invariant Training"
    )
    args = parser.parse_args()
    model = ConvTasNet(
        args.N,
        args.L,
        args.B,
        args.H,
        args.P,
        args.X,
        args.R,
        args.C,
        norm_type=args.norm_type,
        causal=args.causal,
        mask_nonlinear=args.mask_nonlinear,
    )
    q = np.random.randn(1, 32000).astype(np.float32)
    u = mindspore.Tensor.from_numpy(q)
    out = model(u)
    print("*" * 100)
