<div align="center">


# MindAudio

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/mindspore-lab/mindaudio/ut_test.yaml)
![GitHub issues](https://img.shields.io/github/issues/mindspore-lab/mindaudio)
![GitHub pull requests](https://img.shields.io/github/issues-pr/mindspore-lab/mindaudio)
![GitHub](https://img.shields.io/github/license/mindspore-lab/mindaudio)](<img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindaudio">)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://pycqa.github.io/isort/)

[English](README.md) | 中文

[简介](#简介) |
[安装](#安装) |
[快速入门](#快速入门) |
[任务列表](#任务列表) |
[日志](#日志)

</div>

## 简介

MindAudio是一个基于全场景AI框架 [MindSpore](https://www.mindspore.cn/)
建立的，针对语音领域的开源一体化工具包。它提供语音领域的常用音频数据处理、音频特征提取以及音频数据增强等系列API，用户可便利地进行数据预处理；提供常用数据集以及SoTA模型，支持多个语音处理任务如语音识别、文字到语音生成、声纹识别、语音分离等。

<details open>
<summary> 主要特性 </summary>


- **丰富的数据处理API** MindAudio提供了大量易用的数据处理API，用户可轻松实现音频数据分析，以及对音频算法任务中的数据进行特征提取和增强等。

```python
>>> import mindaudio
# 读取音频文件
>>> test_data, sr = mindaudio.read(data_path)
# 对原始数据进行变速
>>> matrix = mindaudio.speed_perturb(signal, orig_freq=16000, speeds=[90,  100])
```

- **集成常用数据集，一键进行数据预处理** 由于音频深度学习领域中数据集较多，处理过程较复杂，对新手不友好。MindAudio针对不同数据提供一套高效的数据处理方案，并支持用户根据需求进行定制化修改。

```python
>>> from ..librispeech import create_base_dataset, train_data_pipeline
# 创建基础数据集
>>>ds_train = create_base_dataset(manifest_filepath，labels)
# 进行数据特征提取
>>>ds_train = train_data_pipeline(ds_train, batch_size=64)
```

- **支持多种任务模型** MindAudio提供多种任务模型, 如ASR任务中的DeepSpeech2，TTS任务中的WavGrad等，并提供预训练权重、训练策略和性能报告，帮助用户快速上手复现音频领域任务。
- **灵活高效** MindAudio基于高效的深度学习框架MindSpore开发，具有自动并行和自动微分等特性，支持不同硬件平台上（CPU/GPU/Ascend），同时支持效率优化的静态图模式和调试灵活的动态图模式。

## 安装

### PyPI安装

MindAudio的已发布版本可以通过PyPI安装。

```shell
pip install mindaudio
```

### 源码安装

Git上最新的MindAudio可以通过以下指令安装。

```shell
git clone https://github.com/mindspore-lab/mindaudio.git
cd mindaudio
pip install -r requirements.txt
python setup.py install
```

## 快速入门

### 音频数据分析

mindaudio提供一系列常用的音频数据处理API，可便捷调用进行数据分析及特征提取。

```python
>>> import mindaudio
>>> import numpy as np
>>> import matplotlib.pyplot as plt
# 读取音频文件
>>> test_data, sr = mindaudio.read(data_path)
# 进行数据特征提取
>>> n_fft = 512
>>> matrix = mindaudio.stft(test_data, n_fft=n_fft)
>>> magnitude, _ = mindaudio.magphase(matrix, 1)
# 画图展示
>>> x = [i for i in range(0, 256*750, 256)]
>>> f = [i/n_fft * sr for i in range(0, int(n_fft/2+1))]
>>> plt.pcolormesh(x,f,magnitude, shading='gouraud', vmin=0, vmax=np.percentile(magnitude, 98))
>>> plt.title('STFT Magnitude')
>>> plt.ylabel('Frequency [Hz]')
>>> plt.xlabel('Time [sec]')
>>> plt.show()
```

结果展示：
![image-20230310165349460](https://raw.githubusercontent.com/mindspore-lab/mindaudio/main/tests/result/stft_magnitude.png)



### 语音任务实现

针对不同的数据集和任务，我们提供不同的数据集预处理和训练策略。以ASR任务为例，我们提供了在LibriSpeech数据集上的训练以及推理示例：

- 数据集准备

```shell
# 进入相应数据集目录
cd recipes/LibriSpeech
# 数据集准备
python librispeech.py --root_path "your_data_path"
```

- 单卡训练

```shell
# 进入具体任务目录
cd ASR
# 单卡训练
python train.py -c "./hparams/Deepspeech2.yaml"
```

- 多卡训练

```shell
# 进入具体任务目录
cd ASR
# 启动多卡训练
mpirun --allow-run-as-root -n 8 python train.py -c "./hparams/Deepspeech2.yaml"
```

- 推理

```shell
#推理
python eval.py -c "./hparams/Deepspeech2.yaml"
```



## 任务列表

目前，MindAudio支持以下任务以及模型。

| 数据集      | 任务                 | 模型        |
| ----------- | -------------------- | ----------- |
| LibriSpeech | Speech Recognition   | DeepSpeech2 |
| AISHELL     | Speech Recognition   | conformer   |
| VoxCeleb    | Speaker Verification | ECAPA-TDNN  |
| LJSpeech    | Text-to-Speech       | Fastspeech2 |
| LJSpeech    | Text-to-Speech       | WaveGrad    |

### 贡献方式

动态版本仍在开发中，如果您发现任何问题或对新功能有任何想法，请通过[issue](https://github.com/mindlab-ai/mindaudio/issues)与我们联系。

### 许可证

本项目遵循[Apache License 2.0](License.md)开源协议。

### 引用

如果你觉得MindAudio对你的项目有帮助，请考虑引用：

```latex
@misc{MindSpore Audio 2022,
    title={{MindSpore Audio}:MindSpore Audio Toolbox and Benchmark},
    author={MindSpore Audio Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindaudio/}},
    year={2022}
}
```
