# FastSpeech2

[FastSpeech2](https://arxiv.org/abs/2006.04558), a text-to-melspectrogram model for text-to-speech systems.

## Demo

## **Dependencies**

1. `pip install -r requirements.txt`
2. Install [MindSpore](https://www.mindspore.cn/install).
3. (Optional) Install `mpirun` for distributed training.

## Generate from your data

```shell
cd mindaudio/examples/fastspeech2
python generate.py --restore fastspeech2_160k_en_mel128.ckpt --text "hello this is a test sentence"
```

## Pretrained Models

| Model | Dataset | Checkpoint | Total Batch Size | Num Mels | Hardware | MindSpore Version |
| -----| ----- | -----| -----| -----| -----| -----|
| FastSpeech2 (base) | LJSpeech-1.1 | [160k steps](https://download.mindspore.cn/toolkits/mindaudio/fastspeech2/fastspeech2_160k_en_mel128.ckpt) | 32 | 128 | 1 $\times$ Ascend | 1.9.0 |
| FastSpeech2 (base) | AiShell | [coming soon]() | 32 | 128 | 1 $\times$ Ascend | 1.9.0 |

## Train your own model

### Step 0 (Data)

#### 0.0 Download

Download [LJSpeech-1.1](http://keithito.com/LJ-Speech-Dataset/) to `./data/`.
Download [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)

#### 0.1 Align

Download provided MFA alignments from [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).

Put the `TextGrid` to be under your path given in `data_path` in `fastspeech2.yaml`.

#### 0.2 Preprocess

Preprocess data to get a "_wav.npy" and "_feature.npy" for each ".wav" file in your dataset folder. Set your `data_path` and
`manifest_path` in `fastspeech2.yaml`. You can now run the following command:

```python
python preprocess.py
```

### Step 1 (Train)

#### 1.1 Train on local server

Training and model parameters can be set in `examples/fastspeech2/fastspeech2.yaml`.

Train on 1 card:
```shell
python train.py --device_target Ascend --context_mode graph 
```

Train on multiple cards:
```shell
mpirun --allow-run-as-root -n 8 python train.py  --context_mode graph --device_target Ascend --is_distributed True
```

### Implementation details

During training, the original paper forwards text token ids to encoder, then expand encodings by durations to the same length of mels, add energy / pitch embeddings, and finally forwards through decoder. This could cause performance issues as different durations in expansion step would lead to too much use of dynamic graph. Instead we first expand the token ids by duration before feeding to encoder, thus keeping everying fixed shape. This will make the attention in encoder behave differently, and a potential trick here is to modify the positional encoding.

### Acknowlegements

Some repositories that inspired this implementation:
- [FastSpeech2](https://github.com/ming024/FastSpeech2)

### License

GNU General Public License v2.0
