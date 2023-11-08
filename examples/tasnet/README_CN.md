# 使用TasNet进行语音分离



## 介绍

TasNet使用编码器-解码器框架直接在时域中对信号进行建模，并对非负编码器输出执行源分离。该方法去除了频率分解步骤，并将分离问题简化为编码器输出上的源掩码估计，然后由解码器合成。该系统降低了语音分离的计算成本，并显着降低了输出所需的最小延迟。TasNet 适用于需要低功耗、实时实现的应用，例如可听设备和电信设备。

[论文](https://arxiv.org/pdf/1711.00541.pdf): TASNET: TIME-DOMAIN AUDIO SEPARATION NETWORK FOR REAL-TIME, SINGLE-CHANNEL SPEECH SEPARATION

### 模型结构

- encoder：提取语音特征
- separation：将encoder得到的结果传入一个4层的LSTM并进行分离
- decoder：将分离结果进行处理，得到语音波形

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
python train.py -c "tasnet.yaml"
```

注意:默认使用Ascend机器

#### 在Ascend上进行多卡训练

此样例使用 8张NPU，如果你想改变NPU的数量，可以更改下列命令中 -n 后的卡数。
```shell
# Distribute_training
mpirun -n 8 python train.py -c "tasnet.yaml"
```
注意:如果脚本是由root用户执行的，必须在mpirun中添加——allow-run-as-root参数，如下所示:
```shell
mpirun --allow-run-as-root -n 8 python train.py -c "tasnet.yaml"
```

### 3.评估模型

```shell
# Validate a trained model
python eval.py -c "tasnet.yaml"
```



## **模型表现**

| 模型   | 机器     | SI-SNR | 参数                                                         |
| ------ | -------- | ------ | ------------------------------------------------------------ |
| tasnet | D910x8-G | 5.97   | [yaml](https://github.com/mindsporelab/mindaudio/blob/main/example/tasnet/tasnet.yaml) |
