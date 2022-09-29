# Deep Speech 2: End-to-End Speech Recognition in
English and Mandarin

Here's a simple example on using mindaudio + mindspore to reproduce the paper [DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf):

- Preprocess [LibriSpeech](https://gitee.com/link?target=http%3A%2F%2Fwww.openslr.org%2F12)

```shell
python examples/deepspeech2/preprocess_deepspeech2_librispeech.py 
```

- Standalone Training

```shell
python examples/deepspeech2/train.py 
```

- Inference

```shell
python examples/deepspeech2/eval.py 
```
