# WaveGrad

[WaveGrad](https://arxiv.org/abs/2009.00713), a diffusion based vocoder model for text-to-speech systems.

## Demo

[sample](results/1000000_predicted_LJ010-0142_1000.wav) Transcript: "Be this as it may, the weapon used was only an ordinary axe, which rather indicates that force, not skill, was employed.")

![compare_lj](results/1000000_LJ010-0142.gif?raw=true "LJ010-0142")

[sample](results/1000000_predicted_fs_1000.wav) Transcript: "This is a MindSpore implementation of the WaveGrad model, a diffusion based vocoder model for text to speech systems. Many thanks to Open I for computational resources!"):

![compare_fs2](results/1000000_fs.gif?raw=true "fs2")

## **Dependencies**

1. `pip install -r requirements.txt`
2. Install [MindSpore](https://www.mindspore.cn/install).
3. (Optional) Install `mpirun` for distributed training.

## Generate from your data

From wav files:

```shell
cd mindaudio/examples/wavegrad
python reverse.py --restore model_1000000.ckpt --wav LJ010-0142.wav --save results --device_target Ascend --device_id 0
```

From melspectrograms:

`python reverse.py --restore model_1000000.ckpt --mel mel.npy --save results --device_target Ascend --device_id 0`

## Pretrained Models

| Model | Dataset | Checkpoint | Total Batch Size | Num Frames | Num Mels | Hardware | MindSpore Version |
| -----| ----- | -----| -----| -----| -----| -----| -----|
| WaveGrad (base) | LJSpeech-1.1 | [1M steps](https://download.mindspore.cn/toolkits/mindaudio/wavegrad/model_1m_base_v190.ckpt) | 256 | 30 | 128 | 8 $\times$ Ascend | 1.9.0 |
| WaveGrad (base) | AiShell | [coming soon]() | 256 | 30 | 128 | 8 $\times$ Ascend | 1.9.0 |

## Train your own model

### Step 0 (Data)

#### 0.0 Download

Download [LJSpeech-1.1](http://keithito.com/LJ-Speech-Dataset/) to `./data/`.

#### 0.1 Preprocess

Preprocess data to get a "_wav.npy" and "_feature.npy" for each ".wav" file in your dataset folder. Set your `data_path` and
`manifest_path` in `wavegrad_base.yaml`. You can now run the following command:

`python preprocess.py --device_target CPU --device_id 0`

### Step 1 (Train)

#### 1.1 Train on local server

Other training and model parameters can be set in `wavegrad_base.yaml`.

Train on 8 cards: (each card will have a batch size of hparams.batch_size // 8)
```
mpirun --allow-run-as-root -n 8 python train.py --device_target Ascend --is_distributed True --context_mode graph
```

Train on 1 card:
```
python train.py --device_target Ascend --device_id 0 --context_mode graph
```

### Implementation details

The interpolation operator in both downsample and upsample blocks are replaced by a simple repeat operator, then divided by repeat factor.

Some additions in UBlock are divided by a constant $\sqrt{2}$ to avoid potential numerical overflow.

### Acknowlegements

Some materials helpful for understanding diffusion models:
- [by Yang Song](https://www.youtube.com/watch?v=nv-WTeKRLl0)
- [by MIT 6.S192](https://www.youtube.com/watch?v=XCUlnHP1TNM)
- [by Outlier](https://www.youtube.com/watch?v=HoKDTa5jHvg)
- [by LilianWeng](lilianweng.github.io/posts/2021-07-11-diffusion-models)

Some repositories that inspired this implementation:
- [lmnt](https://github.com/lmnt-com/wavegrad)
- [FastSpeech2](https://github.com/ming024/FastSpeech2)

### License

GNU General Public License v2.0
