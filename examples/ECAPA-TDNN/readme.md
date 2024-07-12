# 说话人验证--使用ECAPA-TDNN提取说话人特证



## 介绍

ECAPA-TDNN由比利时哥特大学Desplanques等人于2020年提出，通过引入SE (squeeze-excitation)模块以及通道注意机制，此模型在国际声纹识别比赛（VoxSRC2020）中取得了第一名的成绩。

### 模型结构

针对目前基于x-vector的声纹识别系统中的一些优缺点，ECAPA-TDNN从以下3个方面进行了改进：

**1、依赖于通道和上下文的统计池化**

**2、一维Squeeze-Excitation（挤压激励模块）Res2Blocks**

**3、多层特征聚合及求和**

模型结构如下图：

![tdnn.png](https://github.com/mindspore-lab/mindaudio/blob/main/tests/result/tdnn.png?raw=true)

### 数据处理

- 音频：

  1.特征提取：采用fbank。

  2.数据增强：add_babble, add_noise, add_reverb, drop_chunk, drop_freq, speed_perturb。

     当前使用5倍数据增强（需要2.6T磁盘空间）可以得到当前精度。如果想达到EER(0.8%), 需要50倍数据增强, 只需要把`ecapatdnn.yaml`文件中的超参数 `number_of_epochs` 修改为10即可（ 50倍数据增强需要26T磁盘空间）。

## 使用步骤

### 1. 数据集准备（VoxCeleb1 + VoxCeleb2 ）

Voxceleb2 音频文件是m4a格式的，在送给MindAudio之前，所有文件必须先转换成wav文件格式。数据集准备请参考以下步骤:

1. 下载 Voxceleb1、Voxceleb2数据集
请参考右侧网站上指导下载数据集: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
主要是使用Voxceleb1中分离出来的部分音频数据来计算EER。


2. 将m4a文件转换成wav文件
Voxceleb2 是按m4a格式保存音频文件的，想要在MindAudio中使用它们，需要先把所有的m4a文件转换成wav文件。
此脚本调用ffmpeg来完成此转换(https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830)， 转换操作需要数小时，但只需要转换一次即可。


3. 将所有的wav文件放在wav文件夹下，目录结构 `voxceleb12/wav/id*/*.wav` (e.g, `voxceleb12/wav/id00012/21Uxsk56VDQ/00001.wav`)


4. 将下载的`voxceleb1/vox1_test_wav.zip` 压缩文件放到 voxceleb12目录


5. 解压 voxceleb1 test文件， 在voxceleb2 目录下运行 `unzip vox1_test_wav.zip`命令


6. 拷贝 `voxceleb1/vox1_dev_wav.zip` 文件到voxceleb12目录


7. 解压voxceleb1 dev文件, 在voxceleb12目录下执行 `unzip vox1_dev_wav.zip`命令


8. 同时解压 voxceleb1 dev文件、 test文件到目录 `voxceleb1/`. 解压后格式参考 `voxceleb1/wav/id*/*.wav`.


9. 数据增强需要使用 `rirs_noises.zip` 文件， 可以在此链接中下载:http://www.openslr.org/resources/28/rirs_noises.zip ，下载后把它放入 `voxceleb12/` 目录。

### 2. 训练

#### 单卡

Voxceleb1、Voxceleb2 数据集准备好之后, 可以直接运行下面脚本预处理音频数据、在单卡上训练说话人特证:

单卡训练速度较慢，不推荐使用此种方式

```shell
# Standalone training
python train_speaker_embeddings.py
```


​		  2.Voxceleb1、Voxceleb2 数据集很大,预处理音频数据时间比较长，所以在此使用30个进程同时做音频预处理。

可修改`ecapatdnn.yaml`文件中的`data_process_num`参数进行调试

预处理数据生成后, 可以运行以下代码进行单卡训练:

```shell
# Standalone training with prepared data
python train_speaker_embeddings.py --need_generate_data=False
```

#### 多卡

如果预处理数据已经生成，可以运行下面代码进行分布式多卡训练:

`bash ./run_distribute_train_ascend.sh hccl.json`

hccl.json 文件是使用 hccl 工具生成的,可以参考此文章实现 (https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)。 注意，此脚本中要使用hccl.json文件的绝对路径。

### 3.评估

模型训练完成后, 可运行以下脚本做验证:

```shell
# eval
python speaker_verification_cosine.py
```

如果已做过数据预处理, 可设置--need_generate_data=False:

```shell
# eval with prepared data
python speaker_verification_cosine.py --need_generate_data=False
```



## **性能表现**
 - tested on ascend 910 with 8 cards. 
 - total training time : 24hours

| model      | eer with s-norm | eer s-norm | config| weights|
| :-: | :-: | :-: | :-: | :-:|
| ECAPA-TDNN  | 1.50%           | 1.70%      | [yaml](https://github.com/mindsporelab/mindaudio/blob/main/example/ECAPA-TDNN/ecapatdnn.yaml) | [weights](https://download.mindspore.cn/toolkits/mindaudio/ecapatdnn/ecapatdnn_vox12.ckpt) |
