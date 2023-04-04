import mindspore.nn as nn

from mindaudio.models.fastspeech2.loss import FastSpeech2Loss
from mindaudio.models.fastspeech2.utils import get_mask_from_lengths
from mindaudio.models.fastspeech2.variance_adapter import VarianceAdaptor
from mindaudio.models.transformer.models import Decoder, Encoder


class FastSpeech2(nn.Cell):
    def __init__(self, hps):
        super().__init__()
        self.encoder = Encoder(hps)
        self.variance_adaptor = VarianceAdaptor(hps)
        self.decoder = Decoder(hps)
        self.mel_linear = nn.Dense(hps.model.transformer.decoder_hidden, hps.n_mels)
        self.speaker_emb = None

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        positions_encoder=None,
        positions_decoder=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks, positions_encoder)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            _,
            mel_mask,
        ) = self.variance_adaptor(
            x=output,
            src_mask=src_masks,
            mel_mask=mel_masks,
            max_len=None if max_mel_len is None else int(max_mel_len.asnumpy()),
            pitch_target=p_targets,
            energy_target=e_targets,
            duration_target=d_targets,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )
        output, mel_masks = self.decoder(x, mel_mask, positions_decoder)
        output = self.mel_linear(output)
        return (
            output,
            mel_masks,
            src_masks,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
        )

    def construct(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FastSpeech2WithLoss(FastSpeech2):
    def __init__(self, hps):
        super().__init__(hps)
        self.loss_fn = FastSpeech2Loss(hps)

    def construct(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        positions_encoder,
        positions_decoder,
        mels,
        mel_lens,
        max_mel_len,
        p_targets,
        e_targets,
        d_targets,
    ):
        (
            mel_predictions,
            mel_masks,
            src_masks,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
        ) = self.forward(
            speakers=speakers,
            texts=texts,
            src_lens=src_lens,
            max_src_len=max_src_len,
            positions_encoder=positions_encoder,
            positions_decoder=positions_decoder,
            mel_lens=mel_lens,
            max_mel_len=max_mel_len,
            p_targets=p_targets,
            e_targets=e_targets,
            d_targets=d_targets,
        )
        return self.loss_fn(
            mel_targets=mel_predictions,
            pitch_targets=p_targets,
            energy_targets=e_targets,
            duration_targets=d_targets,
            mel_predictions=mel_predictions,
            pitch_predictions=pitch_predictions,
            energy_predictions=energy_predictions,
            log_duration_predictions=log_duration_predictions,
            src_masks=src_masks,
            mel_masks=mel_masks,
        )
