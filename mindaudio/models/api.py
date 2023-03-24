import os

import mindspore as ms

from mindaudio.config import Config, load_config
from mindaudio.models.deepspeech2 import DeepSpeechModel
from mindaudio.models.fastspeech2 import FastSpeech2, FastSpeech2WithLoss
from mindaudio.models.wavegrad import WaveGrad, WaveGradWithLoss

_registry = {
    "wavegrad": [WaveGrad, WaveGradWithLoss, "models/wavegrad/wavegrad.yaml"],
    "fastspeech2": [
        FastSpeech2,
        FastSpeech2WithLoss,
        "models/fastspeech2/fastspeech2.yaml",
    ],
    "deepspeech2": [
        DeepSpeechModel,
        DeepSpeechModel,
        "models/deepspeech2/deepspeech2.yaml",
    ],
}


def create_model(
    model_name: str,
    config: Config = None,
    checkpoint_path: str = None,
    is_train: bool = False,
):
    model_name = model_name.lower()
    if model_name not in _registry:
        raise NotImplementedError(f"model {model_name} has not been implemented yet.")

    if is_train:
        model = _registry[model_name][1]
    else:
        model = _registry[model_name][0]

    print(f"[mindaudio] creating model {model_name}")
    if config is None:
        config = load_config(_registry[model_name][2])
    model = model(config)

    ckpt = None
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            print(f"[mindaudio] loading model from {checkpoint_path}")
            ckpt = ms.load_checkpoint(checkpoint_path, model, strict_load=True)

    return model, ckpt
