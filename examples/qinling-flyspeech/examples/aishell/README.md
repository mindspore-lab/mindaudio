# Qinling-Flyspeech ASR with Aishell

此示例提供使用[Aishell数据集](http://www.openslr.org/resources/33) 训练以及推理qinling-flyspeech模型的代码。

## 概述

## 本地训练与推理

训练以及推理相关的脚本均在`run.sh`中提供, `run.sh`中分为几个阶段，每个阶段实现不同的功能。

| 阶段        | 功能描述 |
| ------     | ----------------|
|-1| 数据集下载，包括下载以及解压。|
|0 | 数据集准备，包括train集、dev集、test集以及CMVN统计文件。|
|1 | 词典准备。|
|2 | 训练所需format.data 文件准备。|
|3 | 网络训练。|
|4 | 网络910验证。|
|5 | 将网络模型导出为MINDIR格式用于310推理。|
|6 | 网络310推理。运行设备为Ascend310系列。|

通过`stage` 和 `stop_stage`参数可以控制执行的阶段数。比如想要执行-1到4所有阶段，命令如下：

```bash
bash run.sh --stage -1 --stop_stage 4
```

当`stage` 和 `stop_stage`参数值相等时，表示只执行某一个阶段。比如只执行数据下载，命令如下：

```bash
bash run.sh --stage -1 --stop_stage -1
```

`run.sh`中定义的变量值均可以通过命令传参的方式进行重新赋值。比如多卡训练时，可以使用如下命令：

```bash
bash run.sh --stage 3 --stop_stage 3 --is_distribute True --device_num 8 --rank_table_file hccl_8p.json
```

用户可以在`run.sh`中直接更改参数值，也可以通过命令传参的方式进行更改。

## 阶段-1

此阶段为数据下载阶段、数据将被下载至自定义目录（即`run.sh`中的`data`参数值)。下载下来的原始数据为压缩文件
data_aishell.tgz与resouce_aishell.tgz。之后程序会自动进行解压。

```bash
bash run.sh --stage -1 --stop_stage -1 --data ${DIR_TO_DOWNLOAD}
```

如您自行下载了数据集，可以手动对 data_aishell.tgz 与 resouce_aishell.tgz 进行解压缩：

```shell
tar zxvf data_aishell.tgz
tar zxvf resouce_aishell.tgz
```

手动解压缩后，data_aishell 文件夹内包含若干压缩包，使用一下命令进行批量解压：

```shell
cd data_aishell/wav
ls *.tar.gz | xargs -n1 tar xzvf
```

## 阶段0

此阶段准备训练以及推理需要的train集、dev集、test集、CMVN统计文件。准备好的数据文件将被放置于data/{train,dev,test}
目录下。其中CMVN文件存放位置为data/train/global_cmvn。

*注意*：${DIR_TO_DOWNLOAD}应为`阶段-1`中自定义下载目录的绝对路径.

```bash
bash run.sh --stage 0 --stop_stage 0 --data ${DIR_TO_DOWNLOAD}
```

## 阶段1

此阶段为词典准备阶段，在 qinling-flyspeech 工程目录下直接运行下面的 shell 脚本。准备字典文件。指定用于指明模型训练过程中的建模单元，针对中文数据集，一般使用字级别建模单元；针对英文数据集，一般使用字母或者 SentencePiece 作为建模单元。准备好的字典文件将被放置于 `data/dict` 目录下。

```bash
bash run.sh --stage 1 --stop_stage 1
```

## 阶段2

此阶段为准备format.data文件。在`qinling-flyspeech`工程目录下直接运行下面的 shell 脚本。针对每个数据集准备 `format.data` 文件用于训练。`format.data` 文件由五列组成，分别是：句子 ID（`utt`），特征或音频路径 （`feat`），特征长度 （`feat_shape`），原始抄本 （`text`），按照字典分成不同建模单元的抄本（`token`）和对应到不同建模单元 ID 的抄本（`tokenid`）。`format.data` 文件将作为直接输入用于模型训练。

```bash
bash run.sh --stage 2 --stop_stage 2
```

## 阶段3

此阶段为训练阶段，分为单卡训练和多卡训练。`run.sh`中的`is_distribute`为`False`时为单卡训练，为`True`时为多卡训练。

单卡训练：
程序会在`qinling-flyspeech`工程目录下创建`${exp}/${net_name}`文件夹，训练生成的`.ckpt`文件将被保存在`${exp}/${net_name}/${exp_name}/model`文件夹下. 日志文件保存在${exp}/${net_name}/train.log中。

多卡训练：
在运行网络之前，准备分布式运行配置文件hccl.json文件。
程序会在`qinling-flyspeech`工程目录下创建`${distribute_dir_name}${device_id}/${exp_name}`文件夹，训练生成的`.ckpt`文件将被保存在`${distribute_dir_name}0/${exp_name}/model`文件夹下。日志文件保存在`${distribute_dir_name}${device_id}/train.log`中。多卡训练时，需要准备rank_table文件，通过`run.sh`中的`rank_table_file`指定。rank_table文件推荐使用脚本生成，点击 [链接]( https://gitee.com/mindspore/models/tree/master/utils/hccl_tools ) 可获取脚本。更多详细信息请参考 [配置分布式环境变量](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/parallel/train_ascend.html#%E9%85%8D%E7%BD%AE%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F) 。

```bash
bash run.sh --stage 3 --stop_stage 3
```

## 阶段4

此阶段为网络验证阶段，提供了四种解码方法：`ctc_greedy_search`、`ctc_prefix_beam_search`、`attention`、`attention_rescoring`。 由于各种解码方法彼此独立，因此程序默认采用多进程异步解码以提高整体解码速度。解码时需要保证有四张空闲卡可用，即一张卡对应于一个进程。

解码时，推荐使用平均模型来解码。当`run.sh`中的`training_with_eval`为`True`时，即训练为边训练边推理，训练完后程序会自动进行模型权重平均化，然后保存最终的模型权重。否则需要调用`tools/average_model.py`计算平均模型。模型平均数默认为30，即对最后的30个权重模型进行平均。

解码结果保存在{exp_predict}/{net_name}/default/test_{mode}/result.txt中。计算CER的结果保存在{exp_predict}/{net_name}/default/test_{mode}/cer.txt中。

```bash
bash run.sh --stage 4 --stop_stage 4
```

## 阶段5

此阶段使用[mindspore.export接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.export.html?highlight=mindspore.export) 将MindSpore模型导出为MINDIR格式，用于Ascend310推理。针对四种解码方式，会分别生成对应的MINDIR格式文件。

```bash
bash run.sh --stage 5 --stop_stage 5
```

## 阶段6

此阶段为Ascend310端侧推理阶段，运行环境为Ascend310，使用MindSpore Lite提供的[Python API](https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite.html) 进行模型转换和模型推理。首先，调用[mindspore_lite.Converter接口](https://www.mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Converter.html#mindspore_lite.Converter) 将MINDIR模型转为端侧MS模型，然后加载MS模型进行端侧推理。

```bash
bash run.sh --stage 6 --stop_stage 6
```

## ModelArts上训练

1.在 ``config/*.yaml``中配置ModelArts参数：

```bash
设置enable_modelarts=True
设置OBS数据集路径data_url: "obs://speech/corpus/aishell1"
设置OBS训练回传路径train_url: "obs://speech/code/asr/workspace/"
```

2.上传aishell1数据集到上述OBS路径``"obs://speech/corpus/aishell1"``

3.上传训练代码到OBS桶内``"obs://speech/code/asr"``

4.登录ModelArts控制台，进入``训练管理/训练作业``页面，选择创建作业

```bash
1）置训练作业名称，运行版本默认为V0001*
2）算法来源配置
    如选择常用框架，则选择AI引擎，设置代码目录，从代码目录中选择启动文件。
    如选择自定义，则根据MindSpore版本要求制作镜像，设置镜像地址及代码目录，设置运行命令如下：
    /bin/bash /home/work/run_train.sh 's3://speech/code/asr' 'asr/train.py' '/tmp/log/train.log' --'data_url'='obs://speech/corpus/aishell1' --'train_url'='obs://speech/code/asr/workspace/'
3）数据来源选择上述OBS数据集路径``"obs://speech/corpus/aishell1"``
4）训练输出位置选择上述OBS训练回传路径``"obs://speech/code/asr/workspace/"``
5）根据实际资源情况，选择公共资源池或专属资源池，配置规格，计算节点参数，开始训练。
```
