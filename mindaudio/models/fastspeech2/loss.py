import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class FastSpeech2Loss(nn.Cell):
    def __init__(self, hps):
        super().__init__()
        self.pitch_feature_level = hps.pitch.feature
        self.energy_feature_level = hps.energy.feature
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.slr = ops.ScalarSummary()
        self.names = ['total_loss', 'mel_loss', 'duration_loss', 'pitch_loss', 'energy_loss']

    def construct(self, items):
        mel_targets = items['mel_targets']
        pitch_targets = items['pitch_targets']
        energy_targets = items['energy_targets']
        duration_targets = items['duration_targets']
        mel_predictions = items['mel_predictions']
        pitch_predictions = items['pitch_predictions']
        energy_predictions = items['energy_predictions']
        log_duration_predictions = items['log_duration_predictions']
        src_masks = items['src_masks']
        mel_masks = items['mel_masks']

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = ms.ops.log(duration_targets + 1.)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions * src_masks
            pitch_targets = pitch_targets * src_masks
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions * mel_masks
            pitch_targets = pitch_targets * mel_masks
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions * src_masks
            energy_targets = energy_targets * src_masks
        elif self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions * mel_masks
            energy_targets = energy_targets * mel_masks
        energy_loss = self.mse_loss(energy_predictions, energy_targets)

        log_duration_predictions = log_duration_predictions * src_masks
        log_duration_targets = log_duration_targets * src_masks
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        mel_masks = mel_masks.expand_dims(-1)
        mel_loss = self.mae_loss(mel_predictions * mel_masks, mel_targets * mel_masks)

        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss
        losses = total_loss, mel_loss, duration_loss, pitch_loss, energy_loss
        for i in range(len(self.names)):
            self.slr(self.names[i], losses[i])

        return losses
