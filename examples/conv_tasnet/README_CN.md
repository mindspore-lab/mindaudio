# 使用Conv-TasNet进行语音分离



## 介绍

全卷积时域音频分离网络（Conv-TasNet）由三个处理阶段组成, 编码器、分离和解码器。首先，编码器模块用于将混合波形的短段转换为它们在中间特征空间中的对应表示。然后，该表示用于在每个时间步估计每个源的乘法函数（掩码）。然后，利用解码器模块对屏蔽编码器特征进行变换，重构源波形
Conv-Tasnet被广泛的应用在语音分离等任务上，取得了显著的效果

[论文](https://arxiv.org/abs/1809.07454): Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation

### 模型结构

模型包括
encoder：类似fft，提取语音特征。
decoder：类似ifft，得到语音波形
separation：类似得到mask，通过mix*单个语音的mask，类似得到单个语音的一个语谱图。通过decoder还原出语音波形。

### 数据处理

- 使用的数据集为: [librimix](<https://catalog.ldc.upenn.edu/docs/LDC93S1/TIMIT.html>)，LibriMix 是一个开源数据集，用于在嘈杂环境中进行源代码分离。
  要生成 LibriMix，请参照开源项目：https://github.com/JorisCos/LibriMix




## 使用步骤

### 1. 数据集准备
数据预处理运行示例:

```text
python preprocess.py
```

### 2. 训练
#### 单卡训练
由于数据集较大，不推荐使用此种训练方式
```shell
# Standalone training
python train.py -c "conv_tasnet.yaml"
```

注意:默认使用Ascend机器

#### 在Ascend上进行多卡训练

此样例使用 8张NPU，如果你想改变NPU的数量，可以更改下列命令中 -n 后的卡数。
```shell
# Distribute_training
mpirun -n 8 python train.py -c "conv_tasnet.yaml"
```
注意:如果脚本是由root用户执行的，必须在mpirun中添加——allow-run-as-root参数，如下所示:
```shell
mpirun --allow-run-as-root -n 8 python train.py -c "conv_tasnet.yaml"
```

也可以使用如下脚本启动分布式训练：
```shell
bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE]
```

### 3.评估模型

```shell
# Validate a trained model
python eval.py -c "conv_tasnet.yaml"
```



## **模型表现**

| 模型        | 机器     | SI-SNR | 参数                                                         |
| ----------- | -------- | ------ | ------------------------------------------------------------ |
| conv_tasnet | D910x8-G | 12.59  | [yaml](https://github.com/mindsporelab/mindaudio/blob/main/example/conv_tasnet/conv_tasnet.yaml) |
