<div align="center">


# MindAudio

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/mindspore-lab/mindaudio/ut_test.yaml)
![GitHub issues](https://img.shields.io/github/issues/mindspore-lab/mindaudio)
![GitHub pull requests](https://img.shields.io/github/issues-pr/mindspore-lab/mindaudio)
![GitHub](https://img.shields.io/github/license/mindspore-lab/mindaudio)](<img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindaudio">)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

English | [中文](README_CN.md)

[Introduction](#introduction) |
[Installation](#installation) |
[Get Started](#get-started) |
[Model List](#model-list) |
[Notes](#notes)

</div>

## Introduction

MindAudio is an open source all-in-one toolkit for the voice field, witch based on the whole scene AI frame: [MindSpore](https://www.mindspore.cn/). MindAudio provides a series of API for common audio data processing, audio feature extraction and audio data enhancement in the field of speech, so that users can preprocess data conveniently. Provides common data sets and SoTA models to support multiple speech processing tasks such as speech recognition, text-to-speech generation, voice print recognition, speech separation, etc.

<details open>
<summary> Major Features </summary>



- **Bulk data handling API.**

  MindAudio provides a large number of easy-to-use data processing API, users can easily achieve audio data analysis, as well as audio algorithm tasks in the data feature extraction and enhancement.

```python
>>> import mindaudio
# Read audio file
>>> test_data, sr = mindaudio.read(data_path)
#Variable speed processing of raw data
>>> matrix = mindaudio.speed_perturb(signal, orig_freq=16000, speeds=[90,  100])
```

- **Integration of common data sets，one-key data preprocessing.**

  Due to the large number of data sets in the audio deep learning field, the processing process is more complex and not friendly to novices. MindAudio provides a set of efficient data processing solutions for different data, and supports users to customize according to their needs.

```python
>>> from ..librispeech import create_base_dataset, train_data_pipeline
# Create the underlying data set
>>>ds_train = create_base_dataset(manifest_filepath，labels)
# Carry out data feature extraction
>>>ds_train = train_data_pipeline(ds_train, batch_size=64)
```

- **Support for multiple task models.**

  MindAudio offers multiple task models, such as DeepSpeech2 in ASR tasks, WavGrad in TTS, and more, along with pre-training weights, training strategies, and performance reports to help iterate audio domain tasks quickly.

- **Flexibility and efficiency.**

  MindAudio is built on MindSpore which is an efficent DL framework that can be run on different hardware platforms (GPU/CPU/Ascend). It supports both graph mode for high efficiency and pynative mode for flexibility.

## Installation

### Install with PyPI

The released version of MindAudio can be installed via `PyPI` as follows:

```shell
pip install mindaudio
```

### Install from Source

The latest version of MindAudio can be installed as follows:

```shell
git clone https://github.com/mindspore-lab/mindaudio.git
cd mindaudio
pip install -r requirements.txt
python setup.py install
```

## Get Started

### Audio data analysis

mindaudio provides a series of commonly used audio data processing apis, which can be easily invoked for data analysis and feature extraction.

```python
>>> import mindaudio
>>> import numpy as np
>>> import matplotlib.pyplot as plt
# Read audio file
>>> test_data, sr = mindaudio.read(data_path)
# Carry out data feature extraction
>>> n_fft = 512
>>> matrix = mindaudio.stft(test_data, n_fft=n_fft)
>>> magnitude, _ = mindaudio.magphase(matrix, 1)
# Drawing display
>>> x = [i for i in range(0, 256*750, 256)]
>>> f = [i/n_fft * sr for i in range(0, int(n_fft/2+1))]
>>> plt.pcolormesh(x,f,magnitude, shading='gouraud', vmin=0, vmax=np.percentile(magnitude, 98))
>>> plt.title('STFT Magnitude')
>>> plt.ylabel('Frequency [Hz]')
>>> plt.xlabel('Time [sec]')
>>> plt.show()
```

Result presentation:

![image-20230310165349460](https://raw.githubusercontent.com/mindspore-lab/mindaudio/main/tests/result/stft_magnitude.png)

### Voice task implementation

For different data sets and tasks, we provide different data set preprocessing and training strategies. Taking ASR task as an example, we provide training and reasoning examples on LibriSpeech dataset:

- Dataset preparation

```shell
# Enter the corresponding data set directory
cd recipes/LibriSpeech
# Dataset preparation
python librispeech.py --root_path "your_data_path"
```

- Standalone training

```shell
# Enter the specific task directory
cd ASR
# Standalone training
python train.py -c "./deepspeech2.yaml"
```

- Distribute training

```shell
# Enter the specific task directory
cd ASR
# Distribute training
mpirun --allow-run-as-root -n 8 python train.py -c "./deepspeech2.yaml"
```

- Validation

```shell
# Validate a trained checkpoint
python eval.py -c "./deepspeech2.yaml"
```



## Task List

Currently，the following tasks and models are supported on MindAudio。

| dataset     | task                 | model       |
| ----------- | -------------------- | ----------- |
| LibriSpeech | Speech Recognition   | DeepSpeech2 |
| AISHELL     | Speech Recognition   | conformer   |
| VoxCeleb    | Speaker Verification | ECAPA-TDNN  |
| LJSpeech    | Text-to-Speech       | Fastspeech2 |
| LJSpeech    | Text-to-Speech       | WaveGrad    |

## Notes

### What's New

- 2022/09/30: beta, 33 data APIs + 3 models
- 2023/03/30: version 0.1.0, 52 dataAPIs + 5 models

### Contributing

We appreciate all contributions to improve MindSpore Audio. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

### License

This project is released under the [Apache License 2.0](LICENSE).

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
