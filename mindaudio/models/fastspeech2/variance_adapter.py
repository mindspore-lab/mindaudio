import mindspore as ms
import mindspore.nn as nn
import numpy as np

from mindaudio.models.fastspeech2.utils import get_mask_from_lengths, pad


class LengthRegulator(nn.Cell):
    def __init__(self):
        super().__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        output = pad(output, max_len)

        return output, ms.Tensor(np.stack(mel_len))

    def expand(self, batch, predicted):
        out = batch.repeat(predicted.asnumpy().astype(np.int32).tolist(), 0)
        return out

    def construct(self, x, duration, max_len):
        # x: [b, t, c]
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Cell):
    def __init__(self, hps):
        super().__init__()

        self.input_size = hps.model.transformer.encoder_hidden
        self.filter_size = hps.model.variance_predictor.filter_size
        self.kernel = hps.model.variance_predictor.kernel_size
        self.conv_output_size = hps.model.variance_predictor.filter_size
        self.dropout = hps.model.variance_predictor.dropout

        self.conv1 = nn.SequentialCell(
            nn.Conv1d(
                self.input_size,
                self.filter_size,
                self.kernel,
                has_bias=True,
                pad_mode="same",
            ),
            nn.ReLU(),
        )
        self.norm1 = nn.LayerNorm((self.filter_size,))
        self.dropout1 = nn.Dropout(keep_prob=1.0 - self.dropout)
        self.conv2 = nn.SequentialCell(
            nn.Conv1d(
                self.filter_size,
                self.filter_size,
                self.kernel,
                has_bias=True,
                pad_mode="same",
            ),
            nn.ReLU(),
        )
        self.norm2 = nn.LayerNorm((self.filter_size,))
        self.dropout2 = nn.Dropout(keep_prob=1.0 - self.dropout)
        self.linear_layer = nn.Dense(self.conv_output_size, 1)

    def construct(self, x, mask=None):
        # x: [b, t, c] from MHA
        x = self.conv1(x.transpose([0, 2, 1])).transpose([0, 2, 1])
        x = self.norm1(x)
        x = self.dropout1(x)

        x = self.conv2(x.transpose([0, 2, 1])).transpose([0, 2, 1])
        x = self.norm2(x)
        x = self.dropout2(x)

        x = self.linear_layer(x)
        x = x.squeeze(-1)

        if mask is not None:
            x *= 1 - mask

        return x


class VarianceAdaptor(nn.Cell):
    def __init__(self, hps):
        super().__init__()
        self.duration_predictor = VariancePredictor(hps)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(hps)
        self.energy_predictor = VariancePredictor(hps)
        self.pitch_feature_level = hps.pitch.feature
        self.energy_feature_level = hps.energy.feature

        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = hps.model.variance_embedding.pitch_quantization
        energy_quantization = hps.model.variance_embedding.energy_quantization
        n_bins = hps.model.variance_embedding.n_bins
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        pitch_min, pitch_max = hps.pitch.pitch_min, hps.pitch.pitch_max
        energy_min, energy_max = hps.energy.energy_min, hps.pitch.pitch_max

        if pitch_quantization == "log":
            self.pitch_bins = ms.Parameter(
                np.exp(
                    np.linspace(
                        np.log(pitch_min + 1e-5), np.log(pitch_max + 1e-5), n_bins - 1
                    )
                ),
                name="pitch_bins_log",
                requires_grad=False,
            )
        else:
            self.pitch_bins = ms.Parameter(
                np.linspace(pitch_min, pitch_max, n_bins - 1),
                name="pitch_bins",
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = ms.Parameter(
                np.exp(np.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)),
                name="energy_bins_log",
                requires_grad=False,
            )
        else:
            self.energy_bins = ms.Parameter(
                np.linspace(energy_min, energy_max, n_bins - 1),
                name="energy_bins",
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, hps.model.transformer.encoder_hidden
        )
        self.energy_embedding = nn.Embedding(
            n_bins, hps.model.transformer.encoder_hidden
        )

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask) * control
        if target is not None:
            embedding = self.pitch_embedding(self.pitch_bins.searchsorted(target))
        else:
            embedding = self.pitch_embedding(self.pitch_bins.searchsorted(prediction))
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask) * control
        if target is not None:
            embedding = self.energy_embedding(self.energy_bins.searchsorted(target))
        else:
            embedding = self.energy_embedding(self.energy_bins.searchsorted(prediction))
        return prediction, embedding

    def construct(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        # x: [b, t, c]
        log_duration_prediction = self.duration_predictor(x, src_mask)
        pitch_prediction = None
        energy_prediction = None
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            T = ms.ops.round(ms.ops.exp(log_duration_prediction) - 1) * d_control
            duration_rounded = ms.ops.clip_by_value(T, ms.Tensor(0), None)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )
