# Librispeech ASR with DeepSpeech2



## DeepSpeech2

DeepSpeech2 is a speech recognition models which is trained with CTC  loss. It replaces entire pipelines of hand-engineered components with  neural networks and can handle a diverse variety of speech including  noisy environments, accents and different languages.  The repo supports training/testing and inference using the [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) model both on NPU and GPU.

### Model Architecture

The current reproduced model consists of:

- two convolutional layers:
  - number of channels is 32, kernel size is [41, 11], stride is [2, 2]
  - number of channels is 32, kernel size is [41, 11], stride is [2, 1]
- five bidirectional LSTM layers (size is 1024)
- one projection layer (size is number of characters plus 1 for CTC blank symbol, 28)

### Data Processing

- Audio:


1. Feature extraction: use log power spectrum.


2. Data enhancement: not used yet, coming soon.


- Text:


​	Text encoding with labels,  and can be replaced with a tokenizer model if necessary.

## How to run

### 1.Dataset preparation
This process is described in detail in the upper-level directory. Normally, the corresponding json files which including
wav and txt address information，generated in each data folder.

### 2.training
```shell
# Enter the specific task directory
cd ASR
```
#### Standalone training
Because of the large amount of data, it is not recommended to use a single card for training.
```shell
# Standalone training
python train.py -c "./hparams/deepspeech2.yaml"
```

#### Distribute training on Ascend
This example uses 8 NPUs, if you want to change the quantity, please change the number of cards after -n
```shell
# Distribute_training
mpirun -n 8 python train.py -c "./deepspeech2.yaml"
```
Notes: If the script is executed by the root user, the --allow-run-as-root parameter must be added to mpirun, like this:
```shell
mpirun --allow-run-as-root -n 8 python train.py -c "./deepspeech2.yaml"
```

#### Distribute training on GPU
If you want to use the GPU for distributed training, see the following command：
```shell
# Distribute_training
# assume you have 8 GPUs
mpirun -n 8 python train.py -c "./deepspeech2.yaml" --device_target "GPU"
```

### 3.Validation

```shell
# Validate a trained model
python eval.py -c "./deepspeech2.yaml" --pre_trained_model_path "xx.ckpt"
```



## **Performance**

| Model       | Context  | LM   | Test Clean CER | Test Clean WER | Recipe                                                       | Download                                                     |
| ----------- | -------- | ---- | -------------- | -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DeepSpeech2 | D910x8-G | No   | 3.461          | 10.24          | [yaml](https://github.com/LiTingyu1997/mindaudio/blob/main/recipes/LibriSpeech/ASR/hparams/deepspeech2.yaml) | [weights](https://download.mindspore.cn/toolkits/mindaudio/deepspeech2/deepspeech2.ckpt) |
