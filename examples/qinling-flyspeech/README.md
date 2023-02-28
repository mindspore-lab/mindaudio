# qinling-flyspeech

## 项目介绍

qinling-flyspeech是基于MindSpore开发的训练推理部署端到端语音识别算法工具包，是西北工业大学和华为在探索智能语音道路上的重要成果，中文名称为秦岭·翔语，本仓是秦岭·翔语项目中的ASR模型的代码，软硬件采用全国产华为昇腾全栈。

## 软件架构

qinling-flyspeech主要分为两层，上层是提供直接训练、推理任务的脚本代码；下层是模型构建部分，主要分为配置、数据、模型以及推理部署等部分构成。

 **注**：虚线部分待实现。

![qinling-flyspeech软件架构](docs/source/architecture.png)

## 安装教程

1. 安装 CANN（以 arm + 欧拉的系统配置为例，x86 系统请选择 x86 的包)

    前往 [昇腾社区](https://www.hiascend.com/) 下载安装包 [CANN](https://www.hiascend.com/software/cann/commercial)。

    安装驱动：`./A800-9000-npu-driver_<版本>_linux-aarch64.run --full`
    安装固件：`./A800-9000-npu-firmware_<版本>.run --full`
    安装cann-toolkit包：`./Ascend-cann-toolkit_<版本>_linux-aarch64.run --full`

2. 安装 MindSpore

    MindSpore 最低版本要求为 1.7.1，请前往 [MindSpore](https://mindspore.cn/) 官网，按照教程安装最新版本即可，当前建议安装 MindSpore 1.9 版本。

3. 安装 requirements 依赖：`pip install -r requirements.txt`

## 代码结构

```shell
qinling-flyspeech
|—— examples  # 示例
|    |—— aishell # aishell数据集示例
|    |    |—— config
|    |        |—— asr_conformer.yaml # encoder为conformer结构的asr配置文件
|    |        |—— asr_transformer.yaml # encoder为transformer结构的asr配置文件
|    |    |—— local # 数据集下载及准备脚本（Aishell 数据集）
|    |        |—— aishell_data_prep.sh # 数据格式制作
|    |        |—— download_and_untar.sh # 数据下载
|    |    |—— one_stop_asr_predict.sh # asr单卡训练启动脚本
|    |    |—— one_stop_distribute_asr_train.sh # asr多卡训练启动脚本
|    |    |—— one_stop_standalone_asr_train.sh # asr单卡训练启动脚本
|    |    |—— path.sh
|    |    |—— run.sh
|    |    |—— tools # 软连接到../../tools
|    |    |—— README.md
|    |    |—— RESULT.md # 性能精度记录，包括Ascend910和Ascend310P
|—— flyspeech # 核心代码
|    |—— adapter # 配置以及modelarts适配相关文件
|    |—— dataset # 数据处理
|    |—— decode  # 语音识别推理解码
|    |—— layers  # 模型构成层
|    |—— transformer # transformer以及conformer结构实现
|    |—— model   # 模型实现
|    |—— utils   # 相关实例代码
|—— infer # 推理代码
|    |—— infer_ascend_python # ascend python 推理代码
|—— tools   # CER计算、模型导出等相关工具代码
|—— predict.py # asr推理代码
|—— train.py # asr训练代码
|__ README.md
```

## 使用说明

### 支持模型

qinling-flyspeech支持encoder：transformer，conformer模型。

|   模型类别  |       结构       |
| ---------- | ---------------- |
| ASR        | ASR-transformer  |
|            | ASR-conformer    |

## 免责声明

qinling-flyspeech 仅提供下载和预处理公共数据集的脚本。我们不拥有这些数据集，也不对它们的质量负责或维护。请确保您具有在数据集许可下使用该数据集的权限。在这些数据集上训练的模型仅用于非商业研究和教学目的。

致数据集拥有者：如果您不希望将数据集包含在 qinling-flyspeech 中，或者希望以任何方式对其进行更新，我们将根据要求删除或更新所有公共内容。请通过 Gitee 与我们联系。非常感谢您对这个社区的理解和贡献。

qinling-flyspeech 已获得 Apache 2.0 许可，请参见 LICENSE 文件。

## 许可证

[Apache 2.0许可证](https://gitee.com/mindspore/qinling-flyspeech/blob/master/LICENSE)

## 致谢

qinling-flyspeech借鉴了一些优秀的开源项目，包括

1. 语音识别工具包 [WeNet](https://github.com/wenet-e2e/wenet)
2. Transformer 建模 [ESPnet](https://github.com/espnet/espnet)
3. WFST 解码 [Kaldi](http://kaldi-asr.org/)

## FAQ

优先参考 [ModelZoo FAQ](https://gitee.com/mindspore/models#FAQ) 来查找一些常见的公共问题。

- **Q：数据处理时出现 `No module named bz2` ？**

  **A**：编译 python 时缺少 bz2 库，在安装 bz2 后重新编译 python。

    - Ubuntu：

      ```shell
      sudo apt-get install libbz2-dev
      ```

    - CentOS、EulerOS 和 OpenEuler：

      ```shell
      sudo yum install bzip2
      ```
      
- **Q：训练时第一个epoch部分step耗时超过200s？**

  **A**：由于采用分桶设置，在第一个epoch会针对不同bucket重新编译图，所以部分step会耗时长，属于正常现象。
