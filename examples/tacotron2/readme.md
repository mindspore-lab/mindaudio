# Tacotron2: Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions

Here's a simple example on using mindaudio + mindspore to reproduce the paper [Tacotron2](https://arxiv.org/abs/1712.05884):

- Preprocess LJSpeech

```shell
python examples/tacotron2/preprocess_tacotron2_ljspeech.py --config_path examples/tacotron2/config.yaml
```

- Standalone Training, [config](mindaudio/examples/tacotron2/config.yaml)

```shell
python examples/tacotron2/train.py --config_path examples/tacotron2/config.yaml
```

- Inference

```shell
python examples/tacotron2/eval.py --config_path examples/tacotron2/config.yaml
``` 
