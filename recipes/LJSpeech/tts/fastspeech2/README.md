# FastSpeech2

MindSpore implementation of [FastSpeech2](https://arxiv.org/abs/2006.04558), a text-to-melspectrogram model for text-to-speech systems.

## Demo

[sample](results/1000000_predicted_fs_1000.wav) Transcript: "This is a MindSpore implementation of the FastSpeech2 model, a diffusion based vocoder model for text to speech systems. Many thanks to Open I for computational resources!"):

## **Dependencies**

1. `pip install -r requirements.txt`
2. Install [MindSpore](https://www.mindspore.cn/install).
3. (Optional) Install `mpirun` for distributed training.

## Generate from your data

`python recipes/LJSpeech/tts/fastspeech2/generate.py --restore fastspeech2_160k_en_mel128.ckpt --text "hello this is a test sentence"`

## Pretrained Models

| Model | Dataset | Checkpoint | Total Batch Size | Num Mels | Hardware | MindSpore Version |
| -----| ----- | -----| -----| -----| -----| -----|
| FastSpeech2 (base) | LJSpeech-1.1 | [160k
steps](https://download.mindspore.cn/toolkits/mindaudio/fastspeech2/fastspeech2_160k_en_mel128.ckpt) | 32 | 128 | 1 $\times$ Ascend | 1.9.0 |
| FastSpeech2 (base) | AiShell | [TODO]() | 32 | 128 | 1 $\times$ Ascend | 1.9.0 |

## Train your own model

### Step 0 (Data)

#### 0.0

Download [LJSpeech-1.1](http://keithito.com/LJ-Speech-Dataset/) to `./data/`.
Download [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)

#### 0.1

Download provided MFA alignments from [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).

Put the `TextGrid` to be under your path given in `data_path` in `fastspeech2.yaml`.

#### 0.2

Preprocess data to get a "_wav.npy" and "_feature.npy" for each ".wav" file in your dataset folder. Set your `data_path` and 
`manifest_path` in `fastspeech2.yaml`. You can now run the following command:

```python
python recipes/LJSpeech/tts/fastspeech2/preprocess.py
```

### Step 1 (Train)

#### 1.1 Train on local server

Set up device information:
```
export MY_DEVICE=Ascend # options: [Ascend, GPU]
export MY_DEVICE_NUM=1
```

Other training and model parameters can be set in `recipes/LJSpeech/tts/fastspeech2/fastspeech2.yaml`. 

Train on 1 card:
```
export MY_DEVICE_ID=0
nohup python train.py --device_target $MY_DEVICE --device_id $MY_DEVICE_ID > train_single.log &
```

### Implementation details


### Acknowlegements

Some repositories that inspired this implementation:
- [MindAudio](https://github.com/mindspore-lab/mindaudio)
- [FastSpeech2](https://github.com/ming024/FastSpeech2)

### License

GNU General Public License v2.0
