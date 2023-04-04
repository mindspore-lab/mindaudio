# 语音识别--使用Librispeech数据集
此目录包含了基于 [LibriSpeech](http://www.openslr.org/resources/12)数据集的语音识别任务中，数据集以及需要的文件的准备过程。

[LibriSpeech](http://www.openslr.org/resources/12)数据集是由Vassil  Panayotov编写的16kHz读取英语演讲的语料库，其中包含的960小时的训练数据集，被广泛用于语音识别模型的训练以及效果验证。数据来源于LibriVox项目的阅读有声读物，经过切割和整理成每条10秒左右的音频文件，并进行了相应的文本标注。

### 数据集主要信息

- 训练集：

  - train-clean-100: [6.3G] (100小时的无噪音演讲训练集)
  - train-clean-360.tar.gz [23G] (360小时的无噪音演讲训练集)
  - train-other-500.tar.gz [30G] (500小时的有噪音演讲训练集)

  验证集：

  - dev-clean.tar.gz [337M] (无噪音)
  - dev-other.tar.gz [314M] (有噪音)

  测试集:

  - test-clean.tar.gz [346M] (测试集, 无噪音)
  - test-other.tar.gz [328M] (测试集, 有噪音)



### 数据集准备

```shell
# Enter the corresponding dataset directory
cd recipes/LibriSpeech
```
如为未下载数据集，可使用提供的脚本进行一键下载以及数据准备，如下所示：

```shell
# Download and creat json
python librispeech_prepare.py --root_path "your_data_path"
```
如已下载好压缩文件，请按如下命令操作：
```shell
# creat json
python librispeech_prepare.py --root_path "your_data_path"  --data_ready True
```

LibriSpeech存储flac音频格式的文件。要在MindAudio中使用它们，须将所有flac文件转换为wav文件。

您可以使用[ffmpeg](https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830)或[sox](https://sourceforge.net/projects/sox/)进行转换。该操作可能需要几分钟。

处理后，数据集目录结构如下所示:

```
    ├─ LibriSpeech_dataset
    │  ├── train
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── val
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── test_clean
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  └── test_other
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
```

4个**.json文件存储了相应数据的绝对路径，在后续进行模型训练以及验证中，请将yaml配置文件中的xx_manifest改为对应libri_xx_manifest.json的存放地址。
