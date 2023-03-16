# Librispeech ASR with DeepSpeech2



## How to run

### 1.Dataset preparation

```shell
# Enter the corresponding data set directory
cd recipes/LibriSpeech
# Dataset preparation
python librispeech.py --root_path "your_data_path"
```

Due to reading flac files is not supported, users need to install sox or other software to converse flac to wav files.

### 2.training
#### Standalone training
```shell
# Enter the specific task directory
cd ASR
# Standalone training
python train.py -c "./hparams/Deepspeech2.yaml"
```

#### Distribute training
```shell
# Enter the specific task directory
cd ASR
# Distribute_training
mpirun --allow-run-as-root -n 8 python train.py -c "./hparams/Deepspeech2.yaml"
```

### 3.Validation

```shell
# Validate a trained checkpoint
python eval.py -c "./hparams/Deepspeech2.yaml"
```



## **Performance**

| Params file      | LM   | Test CER | Ascend |
| ---------------- | ---- | -------- | ------ |
| DeepSpeech2.yaml | No   | 3.526    | 910A   |



## **Pretrained Model**

https://download.mindspore.cn/toolkits/mindaudio/deepspeech2/
