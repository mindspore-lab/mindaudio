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

​		文字编码使用dict进行中文编码转换，用户可使用分词模型进行替换。

## 使用步骤

### 1. 数据集准备
该过程在recipes/aishell文件夹中有详细描述，生成的csv文件包括wav文件地址信息以及转换后的中文编码信息。

### 2. 训练
#### 单卡训练
单卡训练速度较慢，不提倡使用此种方式
```shell
# Standalone training
python train.py
```

注意:默认使用Ascend机器

#### 在Ascend上进行多卡训练

此样例使用 8张NPU，如果你想改变NPU的数量，可以更改下列命令中 -n 后的卡数。
```shell
# Distribute_training
mpirun -n 8 python train.py
```
注意:如果脚本是由root用户执行的，必须在mpirun中添加——allow-run-as-root参数，如下所示:
```shell
mpirun --allow-run-as-root -n 8 python train.py
```

#### 在GPU上进行多卡训练
If you want to use the GPU for distributed training, see the following command：
```shell
# Distribute_training
# assume you have 8 GPUs
mpirun -n 8 python train.py  --device_target "GPU"
```

### 3.评估模型

提供ctc greedy search、ctc prefix beam search、attention decoder、attention rescoring四种解码方式，可在yaml配置文件中对解码方式进行修改。

```shell
# Validate a trained model
python eval.py
```



## **模型表现**

* Feature info: using fbank feature, cmvn, online speed perturb
* Training info: lr 0.001, acc_grad 1, 240 epochs, 8 Ascend910
* Decoding info: ctc_weight 0.3, average_num 30
* Performance result: total_time 11h17min

| decoding mode          | CER  |
| ---------------------- | ---- |
| ctc greedy search      | 5.05 |
| ctc prefix beam search | 5.05 |
| attention decoder      | 5.00 |
| attention rescoring    | 4.73 |

