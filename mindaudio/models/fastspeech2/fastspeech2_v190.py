import numpy as np
import mindspore.nn as nn
import mindspore as ms

from mindaudio.models.fastspeech2.utils import get_mask_from_lengths
from mindaudio.models.fastspeech2.variance_adapter import VarianceAdaptor
from mindaudio.models.transformer.models import Decoder, Encoder


class FastSpeech2(nn.Cell):
    def __init__(self, hps):
        super().__init__()
        self.dtype = ms.float16 if hps.use_fp16 else ms.float32
        self.expanded_encoder = Encoder(hps).to_float(self.dtype)
        self.encoder = Encoder(hps).to_float(self.dtype)
        self.variance_adaptor = VarianceAdaptor(hps).to_float(self.dtype)
        self.decoder = Decoder(hps).to_float(self.dtype)
        self.mel_linear = nn.Dense(hps.model.transformer.decoder_hidden, hps.n_mels).to_float(self.dtype)
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
        expanded_phonemes=None,
        expanded_src_lens=None,
        expanded_max_src_len=None,
        src_masks=None,
        mel_masks=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        output = self.encoder(texts, src_masks, positions_encoder)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        yh = self.variance_adaptor(
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
        output, mel_masks = self.decoder(yh['output'], yh['mel_masks'], positions_decoder)
        output = self.mel_linear(output)
        yh.update({
            'mel_predictions': output,
            'mel_masks': mel_masks,
            'src_masks': src_masks,
            'src_lens': src_lens
        })
        return yh

    def forward_expanded(
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
        expanded_phonemes=None,
        expanded_src_lens=None,
        expanded_max_src_len=None,
        src_masks=None,
        mel_masks=None,
    ):
        output = self.encoder(texts, src_masks, positions_encoder)
        # expand, encode, add pitch/energy embedding, decode
        expanded_output = self.expanded_encoder(expanded_phonemes, mel_masks, positions_decoder)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)
            expanded_output = expanded_output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_mel_len, -1)

        yh = self.variance_adaptor.forward_expanded(
            x=output,
            src_mask=src_masks,
            mel_mask=mel_masks,
            max_len=None,
            pitch_target=p_targets,
            energy_target=e_targets,
            duration_target=d_targets,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
            expanded_phonemes=expanded_output,
            expanded_src_lens=expanded_src_lens,
            expanded_max_src_len=expanded_max_src_len,
        )
        output, mel_masks = self.decoder(yh['output'], yh['mel_masks'], positions_decoder)
        output = self.mel_linear(output)
        yh['mel_predictions'] = output
        yh['mel_masks'] = mel_masks
        yh['src_masks'] = src_masks
        yh['src_lens'] = src_lens

        return yh

    def construct(self, *args, **kwargs):
        return self.forward_expanded(*args, **kwargs)

    def infer(
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
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len) if mel_lens is not None else None
        output = self.encoder(texts, src_masks, positions_encoder)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        yh = self.variance_adaptor.infer(
            unexpanded_x=texts,
            x=output,
            src_mask=src_masks,
            mel_mask=mel_masks,
            max_len=None,# if max_mel_len is None else int(max_mel_len.asnumpy()),
            pitch_target=p_targets,
            energy_target=e_targets,
            duration_target=d_targets,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control,
        )
        print("yh['output']:", yh['output'].shape)
        yh['output'] = self.expanded_encoder(yh['output'], mel_masks, positions_decoder)
        print("yh['output']:", yh['output'].shape)
        yh2 = self.variance_adaptor.infer_emb_frame_level(yh['output'], mel_masks, p_control, e_control)
        yh['output'] = yh2['output']
        print("yh['output']:", yh['output'].shape)
        output, mel_masks = self.decoder(yh['output'], yh['mel_masks'], positions_decoder)
        output = self.mel_linear(output)
        yh['mel_predictions'] = output
        yh['mel_masks'] = mel_masks
        yh['src_masks'] = src_masks
        yh['src_lens'] = src_lens
        return yh


class FastSpeech2WithLoss(FastSpeech2):
    def __init__(self, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.total_loss = ms.Parameter(ms.Tensor(0, dtype=self.dtype), requires_grad=False)
        self.mel_loss = ms.Parameter(ms.Tensor(0, dtype=self.dtype), requires_grad=False)
        self.duration_loss = ms.Parameter(ms.Tensor(0, dtype=self.dtype), requires_grad=False)
        self.pitch_loss = ms.Parameter(ms.Tensor(0, dtype=self.dtype), requires_grad=False)
        self.energy_loss = ms.Parameter(ms.Tensor(0, dtype=self.dtype), requires_grad=False)
        self.mel_predictions = ms.Parameter(ms.Tensor(np.ones([1, 742, 128]), dtype=self.dtype), requires_grad=False)

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
        expanded_phonemes=None,
        expanded_src_lens=None,
        expanded_max_src_len=None,
        src_masks=None,
        mel_masks=None,
    ):
        yh = self.forward_expanded(
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
            expanded_phonemes=expanded_phonemes,
            expanded_src_lens=expanded_src_lens,
            expanded_max_src_len=expanded_max_src_len,
            src_masks=src_masks,
            mel_masks=mel_masks,
        )
        yh['mel_targets'] = mels
        yh['pitch_targets'] = p_targets
        yh['energy_targets'] = e_targets
        yh['duration_targets'] = d_targets
        total_loss, mel_loss, duration_loss, pitch_loss, energy_loss = self.loss_fn(yh)
        self.total_loss = total_loss
        self.mel_loss = mel_loss
        self.duration_loss = duration_loss
        self.pitch_loss = pitch_loss
        self.energy_loss = energy_loss
        self.mel_predictions = yh['mel_predictions'][:1]
        return total_loss
