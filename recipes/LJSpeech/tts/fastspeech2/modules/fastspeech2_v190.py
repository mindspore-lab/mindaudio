import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.amp import DynamicLossScaler

import sys
sys.path.append('..')
from modules.transformer.models import Encoder, Decoder
from modules.variance_adapter import VarianceAdaptor
from utils import get_mask_from_lengths


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

    def construct(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FastSpeech2WithLoss(FastSpeech2):
    def __init__(self, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.scale = DynamicLossScaler(1024, 2, 1)

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
        yh = self.forward(
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
        yh.update({
            'mel_targets': mels,
            'pitch_targets': p_targets,
            'energy_targets': e_targets,
            'duration_targets': d_targets,
        })
        return self.scale.scale(self.loss_fn(yh))


if __name__ == '__main__':
    from time import time
    # ms.context.set_context(device_target='GPU', device_id=0, mode=ms.context.GRAPH_MODE)
    ms.context.set_context(device_target='GPU', device_id=0, mode=ms.context.PYNATIVE_MODE)
    args = [
        ms.Tensor(np.load('../speakers.npy')),
        ms.Tensor(np.load('../texts.npy')),
        ms.Tensor(np.load('../src_lens.npy')),
        ms.Tensor(np.load('../max_src_len.npy')),
        # ms.Tensor(np.load('../mels.npy')),
        ms.Tensor(np.load('../mel_lens.npy')),
        ms.Tensor(np.load('../max_mel_len.npy')),
        ms.Tensor(np.load('../p_targets.npy')),
        ms.Tensor(np.load('../e_targets.npy')),
        ms.Tensor(np.load('../d_targets.npy')),
    ]
    print('batch:', args[1].shape)

    import yaml
    preprocess_config = '/home/zhudongyao/ptFastSpeech2/config/LJSpeech_paper/preprocess.yaml'
    model_config = '/home/zhudongyao/ptFastSpeech2/config/LJSpeech_paper/model.yaml'
    train_config = '/home/zhudongyao/ptFastSpeech2/config/LJSpeech_paper/train.yaml'
    preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)

    net = FastSpeech2(preprocess_config, model_config)
    for i in range(2):
        t = time()
        y = net(*args)
        t = time() - t
        print('time: %.2fs' % t)
        for k, v in y:
            print(k, ':', v.shape)