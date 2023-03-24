"""
THIS FILE IS FOR MindSpore 1.9
"""

from math import log as ln

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class Conv1dOrthogonal(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        kwargs["weight_init"] = "Orthogonal"
        kwargs["has_bias"] = True
        super().__init__(*args, **kwargs)


class DBlock(nn.Cell):
    def __init__(self, input_size, hidden_size, factor, kernel_size, dilations):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1dOrthogonal(input_size, hidden_size, 1)
        self.conv = nn.SequentialCell(
            [
                nn.LeakyReLU(0.2),
                Conv1dOrthogonal(
                    input_size,
                    hidden_size,
                    kernel_size[0],
                    dilation=dilations[0],
                    pad_mode="same",
                ),
                nn.LeakyReLU(0.2),
                Conv1dOrthogonal(
                    hidden_size,
                    hidden_size,
                    kernel_size[1],
                    dilation=dilations[1],
                    pad_mode="same",
                ),
                nn.LeakyReLU(0.2),
                Conv1dOrthogonal(
                    hidden_size,
                    hidden_size,
                    kernel_size[2],
                    dilation=dilations[2],
                    pad_mode="same",
                ),
            ]
        )
        self.downscale1 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.factor,
            stride=self.factor,
            pad_mode="valid",
            has_bias=True,
            weight_init="XavierUniform",
        )
        self.downscale2 = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=self.factor,
            stride=self.factor,
            pad_mode="valid",
            has_bias=True,
            weight_init="XavierUniform",
        )

    def construct(self, x):
        residual = self.residual_dense(x)
        residual = self.downscale1(residual)
        x = self.downscale2(x)
        x = self.conv(x)
        return x + residual


class PositionalEncoding(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, x, noise_level):
        count = self.dim // 2
        step = ms.numpy.arange(count, dtype=noise_level.dtype) / count
        encoding = noise_level.expand_dims(1) * ms.ops.exp(
            -ln(1e4) * step.expand_dims(0)
        )
        encoding = ms.ops.concat([ms.ops.sin(encoding), ms.ops.cos(encoding)], -1)
        return x + encoding[:, :, None]


class FiLM(nn.Cell):
    def __init__(self, input_size, output_size, kernel_size):
        super().__init__()
        self.encoding = PositionalEncoding(input_size)
        self.input_conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size,
            weight_init="XavierUniform",
            has_bias=True,
            pad_mode="pad",
            padding=1,
        )
        self.output_conv = nn.Conv1d(
            input_size,
            output_size * 2,
            kernel_size,
            weight_init="XavierUniform",
            has_bias=True,
            pad_mode="pad",
            padding=1,
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

    def construct(self, x, noise_scale):
        x = self.input_conv(x)
        x = self.leaky_relu(x)
        x = self.encoding(x, noise_scale)
        shift, scale = ops.split(self.output_conv(x), 1, output_num=2)
        return shift, scale


class UBlock(nn.Cell):
    def __init__(self, input_size, hidden_size, factor, kernel_size, dilation):
        super().__init__()
        self.factor = factor
        self.block1 = Conv1dOrthogonal(input_size, hidden_size, 1)
        self.block2_a = Conv1dOrthogonal(
            input_size, hidden_size, kernel_size, dilation=dilation[0], pad_mode="same"
        )
        self.block2_b = Conv1dOrthogonal(
            hidden_size, hidden_size, kernel_size, dilation=dilation[1], pad_mode="same"
        )
        self.block3_a = Conv1dOrthogonal(
            hidden_size, hidden_size, kernel_size, dilation=dilation[2], pad_mode="same"
        )
        self.block3_b = Conv1dOrthogonal(
            hidden_size, hidden_size, kernel_size, dilation=dilation[3], pad_mode="same"
        )
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.const = ms.Tensor(2**0.5, dtype=ms.float32)

    def scale_and_shift(self, x, shift, scale):
        x = (scale * x + shift) / self.const
        x = self.leaky_relu(x)
        return x

    def construct(self, x, film_shift, film_scale):
        block1 = self.block1(x)
        block1 = block1.repeat(self.factor, 2) / self.factor

        block2 = self.leaky_relu(x)
        block2 = block2.repeat(self.factor, 2) / self.factor
        block2 = self.block2_a(block2)

        block2 = self.scale_and_shift(block2, film_shift, film_scale)
        block2 = self.block2_b(block2)

        x = (block1 + block2) / self.const

        block3 = self.scale_and_shift(x, film_shift, film_scale)
        block3 = self.block3_a(block3)

        block3 = self.scale_and_shift(block3, film_shift, film_scale)
        block3 = self.block3_b(block3)

        x = (x + block3) / self.const
        return x


class WaveGrad(nn.Cell):
    def __init__(self, hps):
        super().__init__()
        self.DBlock = nn.CellList()
        hidden_size, kernel_size = (
            hps.dblock.init_conv_channels,
            hps.dblock.init_conv_kernels,
        )
        self.DBlock.append(
            Conv1dOrthogonal(1, hidden_size, kernel_size, pad_mode="same")
        )

        input_size = hidden_size
        for hidden_size, factor in zip(hps.dblock.hidden_size, hps.dblock.factor):
            self.DBlock.append(
                DBlock(
                    input_size,
                    hidden_size,
                    factor,
                    hps.dblock.kernel_size,
                    hps.dblock.dilations,
                )
            )
            input_size = hidden_size

        self.FiLM = nn.CellList()
        input_size = hps.dblock.init_conv_channels
        for output_size, kernel_size in zip(hps.film.output_size, hps.film.kernel_size):
            self.FiLM.append(FiLM(input_size, output_size, kernel_size))
            input_size = output_size

        self.UBlock = nn.CellList()
        input_size = hps.first_conv.hidden_size
        for hidden_size, factor, dilation in zip(
            hps.ublock.hidden_size, hps.ublock.factor, hps.ublock.dilation
        ):
            self.UBlock.append(
                UBlock(
                    input_size, hidden_size, factor, hps.ublock.kernel_size, dilation
                )
            )
            input_size = hidden_size

        self.first_conv = Conv1dOrthogonal(
            hps.n_mels,
            hps.first_conv.hidden_size,
            hps.first_conv.kernel_size,
            pad_mode="same",
        )
        self.last_conv = Conv1dOrthogonal(
            hps.ublock.hidden_size[-1], 1, hps.first_conv.kernel_size, pad_mode="same"
        )

    def forward(self, noisy_audio, noise_scale, spectrogram):
        x = noisy_audio.expand_dims(1)
        downsampled = []
        for layer, film in zip(self.DBlock, self.FiLM):
            x = layer(x)
            downsampled.append(film(x, noise_scale))

        x = self.first_conv(spectrogram)
        for layer, (film_shift, film_scale) in zip(self.UBlock, downsampled[::-1]):
            x = layer(x, film_shift, film_scale)
        x = self.last_conv(x).squeeze(1)
        return x

    def construct(self, noisy_audio, noise_scale, spectrogram):
        return self.forward(noisy_audio, noise_scale, spectrogram)


class WaveGradWithLoss(WaveGrad):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.L1Loss()

    def construct(self, noisy_audio, noise_scale, noise, spectrogram):
        yh = self.forward(noisy_audio, noise_scale, spectrogram)
        return self.loss_fn(yh, noise)
