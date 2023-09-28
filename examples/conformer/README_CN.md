# 使用conformer进行语音识别



## 介绍

conformer是将一种transformer和cnn结合起来，对音频序列进行局部和全局依赖都进行建模的模型。目前基于transformer和卷积神经网络cnn的模型在ASR上已经达到了较好的效果，Transformer能够捕获长序列的依赖和基于内容的全局交互信息，CNN则能够有效利用局部特征，因此针对语音识别问题提出了卷积增强的transformer模型，称为conformer，模型性能优于transformer和cnn。目前提供版本支持在NPU和GPU上使用[conformer](https://arxiv.org/pdf/2102.06657v1.pdf)模型在aishell-1数据集上进行训练/测试和推理。

### 模型结构

Conformer整体结构包括：SpecAug、ConvolutionSubsampling、Linear、Dropout、ConformerBlocks×N，可见如下结构图。

- ConformerBlock结构（N个该结构）：Feed Forward Module、Multi-Head Self Attention Module、Convolution Module、Feed Forward Module、Layernorm。其中每个Module都是前接一个Layernorm后接一个Dropout，且都有残差链连接，残差数据为输入数据本身。

- 马卡龙结构：可以看到ConformerBlock神似马卡龙结构，即两个一样的Feed Forward Module中间夹了Multi-Head Self Attention Module和Convolution。

  ![image-20230310165349460](https://raw.githubusercontent.com/mindspore-lab/mindaudio/main/tests/result/conformer.png)

### 数据处理

- 音频：

  1.特征提取：采用fbank。

  2.数据增强：在线speed_perturb。

- 文字：

​		文字编码使用逐字中文编码转换，用户可使用分词模型进行替换。

## 使用步骤

### 1. 数据集准备

以aishell数据集为例，mindaudio提供下载、生成统计信息的脚本（包含wav文件地址信息以及对应中文信息），执行此脚本会生成train.csv、dev.csv、test.csv三个文件。

```shell
# data_path为存放数据的地址
python mindaudio/data/aishell.py --data_path "/data" --download False
```

如需下载数据， --download True

### 2. 数据预处理

#### 文字部分

根据aishell提供的aishell_transcript_v0.8.txt，生成逐字的编码文件，每个字对应一个id，输出包含编码信息的文件：lang_char.txt。

```shell
cd mindaudio/utils
python text2token.py -s 1 -n 1 "data_path/data_aishell/transcript/aishell_transcript_v0.8.txt" | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${/data_path/lang_char.txt}
```

#### 音频部分

本模型使用了全局cmvn，为提高模型训练效率，在训练前会对数据的特征进行统计，生成包含统计信息的文件：global_cmvn.json。

```shell
cd examples/conformer
python compute_cmvn_stats.py --num_workers 16 --train_config conformer.yaml --in_scp data_path/train.csv --out_cmvn data_path/global_cmvn
```

注意：--num_workers可根据训练设备的核数进行调整

### 3. 训练

#### 单卡训练
```shell
cd examples/conformer
# Standalone training
python train.py --config_path ./conformer.yaml
```

注意:默认使用Ascend机器

#### 在Ascend上进行多卡训练

此样例使用 8张NPU.
```shell
# Distribute_training
mpirun -n 8 python train.py --config_path ./conformer.yaml
```
注意:如果脚本是由root用户执行的，必须在mpirun中添加——allow-run-as-root参数，如下所示:
```shell
mpirun --allow-run-as-root -n 8 python train.py --config_path ./conformer.yaml
```

如在GPU中进行训练，可更改yaml文件中的配置。

### 4.评估模型

提供ctc greedy search、ctc prefix beam search、attention decoder、attention rescoring四种解码方式，可在yaml配置文件中对解码方式进行修改。

执行脚本后将生成包含预测结果的文件：result.txt

```shell
python predict.py --config_path ./conformer.yaml
```



## **模型表现**

* Feature info: using fbank feature, cmvn, online speed perturb
* Training info: lr 0.001, acc_grad 1, 240 epochs, 8 Ascend910
* Decoding info: ctc_weight 0.3, average_num 30
* Performance result: total_time 11h17min, 8p, using hccl_tools.

| model     | decoding mode          | CER  |
| --------- | ---------------------- | ---- |
| conformer | ctc greedy search      | 5.05 |
| conformer | ctc prefix beam search | 5.05 |
| conformer | attention decoder      | 5.00 |
| conformer | attention rescoring    | 4.73 |
