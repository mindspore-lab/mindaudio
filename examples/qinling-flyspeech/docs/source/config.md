# 配置文件

qinling-flyspeech 的配置文件整体采用模块化以及继承设计，抽取公共配置为基础配置，不同模型配置文件独立。

## 配置文件目录结构

基础配置文件分别为 `config/asr/asr_config.yaml`，各模型配置对基础配置进行继承，在执行训练脚本时可指定特定配置文件进行训练。

目录结构如下:

```shell
|—— config                           # 配置文件
|    |—— asr                         # asr任务的配置文件
|    |    |—— asr_config.yaml        # asr基础配置说明
|    |    |—— asr_conformer.yaml     # conformer结构的asr配置说明
|    |    |—— asr_transfomer.yaml    # transformer结构的asr配置说明
|    |    |—— e2e_config.yaml        # end2end的配置说明
```

## 配置文件的修改

在模型配置文件中可对基础配置文件中的指定配置选项进行修改，其余配置继承基础配置。例如：

ASR 的基础配置 `config/asr/asr_config.yaml` 中 encoder 的配置为：

```yaml
encoder: transformer
encoder_conf:
    output_size: 256              # dimension of attention
    attention_heads: 4
    linear_units: 2048            # the number of units of position-wise feed forward
    num_blocks: 12                # the number of encoder blocks
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: "conv2d"         # encoder input type
    normalize_before: True
    feature_norm : True
    pos_enc_layer_type: 'abs_pos'
```

`config/asr/asr_conformer.yaml` 对上述基础配置文件进行继承：

```yaml
# base config
base_config: ['./asr_config.yaml']
```

在`config/asr/asr_conformer.yaml`中可指定 encoder 为 conformer，并修改相关配置：

```yaml
encoder: conformer
encoder_conf:
    output_size: 256                   # dimension of attention
    attention_heads: 4
    linear_units: 2048                 # the number of units of position-wise feed forward
    num_blocks: 12                     # the number of encoder blocks
    dropout_rate: 0.1
    positional_dropout_rate: 0.1
    attention_dropout_rate: 0.0
    input_layer: conv2d                # encoder input type
    normalize_before: True
    cnn_module_kernel: 15
    activation_type: 'swish'
    pos_enc_layer_type: 'rel_pos'
    feature_norm : True
```

## 基础配置说明

modelarts 使用适配几部分。

以 ASR 为例，具体如下：

+ 网络结构

  ```yaml
  # network architecture
  encoder: transformer
  encoder_conf:
      output_size                         attention的输出维度
      attention_heads                     编码层attention head的数量，默认为12
      linear_units                        FeedForward层的隐藏单元数量，默认为2048
      num_blocks                          编码器层数，默认为12
      dropout_rate                        FeedForward层的随机失活率
      positional_dropout_rate             位置编码层的随机失活率
      attention_dropout_rate              attention层的随机失活率
      input_layer                         输入层的类型，默认为conv2d
      normalize_before                    是否在子模块前进行归一化，可选项为True或False，默认为True
      feature_norm                        是否进行特征归一化，可选项为True或False，默认为False
      pos_enc_layer_type                  位置编码类型，可选项为abs_pos,rel_pos或conv_pos,默认为abs_pos

  # decoder related
  decoder: transformer
  decoder_conf:
      attention_heads                     解码层attention head的数量，默认为4
      linear_units                        FeedForward层的隐藏单元数量，默认为2048
      num_blocks                          解码器层数，默认为6
      dropout_rate                        FeedForward层的随机失活率
      positional_dropout_rate             位置编码层的随机失活率
      self_attention_dropout_rate         前attention层的随机失活率
      src_attention_dropout_rate          中间attention层的随机失活率
  ```

+ 特征提取

  ```yaml
  # feature extraction
  collate_conf:
      # feature level config
      feature_extraction_conf:
          feature_type                    特征类型，默认为fbank
          mel_bins                        特征维度，默认为80
          frame_shift                     帧移，默认为10
          frame_length                    帧长，默认为25
      # data augmentation config
      speed_perturb                       速度扰动
      spec_aug                            数据遮掩
      spec_aug_conf:
          num_t_mask                      时域遮掩次数，默认为2
          num_f_mask                      频域遮掩次数，默认为2
          max_t                           时域遮掩最大长度，默认为50
          max_f                           频域遮掩最大长度，默认为10
          max_w                           特征维度
  ```

  ASR 使用 Fbank 提取音频特征，将特征提取到 80 个维度，再对特征进行分帧后进行后续训练。在数据增强方面使用了 speed_perturb 和 spec_aug，speed_perturb 随机对音频进行 0.9 或 1.1 倍变速，spec_aug 在时域和频域随机对数据进行遮掩避免过拟合问题。

+ 数据集

  ```yaml
  # dataset related
  dataset_conf:
      max_length                          数据最大长度
      min_length                          数据最小长度
      token_max_length                    token最大数量
      token_min_length                    token最小数量
      batch_type                          batch类型，可选项为bucket, static 或 dynamic，默认为bucket
      frame_bucket_limit                  分桶数据长度限制
      batch_bucket_limit                  分桶数据批次大小限制
      batch_factor                        批次缩放系数
      shuffle                             是否进行数据打乱，可选项为True或False，默认为True
  ```

  数据集处理方面 asr使用分桶设置，通过设立不同长度的 bucket 对数据进行划分。

  `max_length` 和 `min_length` 对数据长度作出限制，`token_max_length` 和 `token_min_length` 限制音频便签字数，`frame_bucket_limit` 规定各 bucket 内数据长度，`batch_bucket_limit` 规定各 bucket 的 batch size，可根据内存及显卡配置适当减少 bucket 数量，或通过 `batch_factor` 调整 batch size。

+ 训练参数

  ```yaml
  grad_clip                               梯度裁剪
  accum_grad                              梯度累计
  max_epoch                               训练轮次
  log_interval                            输出训练日志的间隔

  optim                                   优化器，默认为adam
  optim_conf:
      lr                                  学习率
  scheduler                               学习率衰减
  scheduler_conf:
      warmup_steps                        warmup步数

  # train option
  exp_name                                训练名称
  train_data                              训练数据集路径
  eval_data                               验证数据集路径
  save_checkpoint                         是否保留ckpt，可选项为True或False，默认为True
  save_checkpoint_epochs                  每多少轮保留ckpt
  keep_checkpoint_max                     最多保留ckpt数量
  save_checkpoint_path                    保存ckpt文件路径
  device_target                           设备
  is_distributed                          是否分布式训练，可选项为True或False，默认为False
  mixed_precision                         是否使用混合精度，可选项为True或False，默认为True
  ckpt_file                               继续训练的ckpt文件路径
  save_graphs                             是否保存计算图，可选项为True或False，默认为False
  training_with_eval                      是否在训练中验证，可选项为True或False，默认为False
  ```

+ 推理参数

  ```yaml
  # decode option
  test_data                               测试数据集路径
  dict                                    字典路径
  decode_ckpt                             用于推理的ckpt文件路径
  decode_mode                             推理方式，可选项为attention，ctc_greedy_search，ctc_prefix_beam_search，默认为attention
  full_graph                              是否全图推理，可选项为True或False，默认为True
  decode_batch_size                       推理批次大小
  ctc_weight                              ctc权重
  beam_size                               集束搜索束宽
  ```

+ modelarts使用适配

  ```yaml
  enable_modelarts                        是否使用modelarts，可选项为True或False，默认为False
  # Url for modelarts
  data_url                                obs数据集路径
  train_url                               obs训练路径
  checkpoint_url                          obs预训练ckpt文件路径
  # Path for local
  data_path                               数据路径
  output_path                             输出路径
  load_path                               ckpt文件路径
  need_modelarts_dataset_unzip            是否需要解压数据集，可选项为True或False，默认为False
  modelarts_dataset_unzip_name            数据集压缩包名称
  ```

+ 其他配置说明

  ```yaml
  enable_profiling                        是否进行性能分析，可选项为True或False，默认为False
  enable_summary                          是否收集训练summary信息，可选项为True或False，默认为False
  ```
