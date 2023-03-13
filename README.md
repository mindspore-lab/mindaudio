# MindAudio

## Introduction

MindSpore Audio is an open source audio research toolbox based on MindSpore in audio direction. Mainly focused on rapid development and implementation for audio task researching, we provide numerous audio processing APIs, deep learning model implementations, as well as example preprocess-train-infer pipeline python scripts for academic purposes. These scripts are designed to be easily adapt to custom research projects.

## Getting Started

### [Conformer](/examples/conformer/)
### [DeepSpeech2](/recipes/LibriSpeech/)
### [WaveGrad](/recipes/LJSpeech/tts/wavegrad)
### [FastSpeech2](/recipes/LJSpeech/tts/fastspeech2)

## API

### [mindaudio.data](/mindaudio/data)
### [mindaudio.models](/mindaudio/models)
### [mindaudio.callbacks](/mindaudio/callbacks)

## Installation

1. Install dependency
```shell
pip install -r requirements.txt
```

2. Install [MindSpore](https://www.mindspore.cn/install)

3. (optional) and [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) for distributed mode.   

4. Install mindaudio

```shell
git clone https://github.com/mindlab-ai/mindaudio.git
cd mindaudio
python setup.py install
# or
pip install .
```

or

```shell
sh package.sh
pip install output/YOUR_PACKAGE.whl
```

## Notes

### What's New 

- 2022/09/30: version 0.1.0, 33 data APIs + 3 models

### License

This project is released under the [Apache License 2.0](LICENSE).

### Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [issue](https://github.com/mindlab-ai/mindaudio/issues).

### Acknowledgement

MindAudio is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new Audio methods.

### Contributing

We appreciate all contributions to improve MindSpore Audio. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Audio 2022,
    title={{MindSpore Audio}:MindSpore Audio Toolbox and Benchmark},
    author={MindSpore Audio Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindaudio/}},
    year={2022}
}
```
