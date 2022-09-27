# Conformer: Convolution-augmented Transformer for Speech Recognition

Here's a simple example on using mindaudio + mindspore to reproduce the paper [Conformer](https://arxiv.org/abs/2005.08100):

- Standalone Training, [config](mindaudio/examples/conformer/asr_conformer.yaml)

```shell
python examples/conformer/train.py --config_path examples/conformer/asr_conformer.yaml
```

- Inference

```shell
python examples/conformer/eval.py --config_path examples/conformer/asr_conformer.yaml
``` 
