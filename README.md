# MindSpore Audio

## Introduction
-----
MindSpore Audio is an open source audio research toolbox based on MindSpore in audio direction. Mainly focused on rapid development and implementation for audio task researching, we provide numerous audio processing APIs, deep learning model implementations, as well as example preprocess-train-infer pipeline python scripts for academic purposes. These scripts are designed to be easily adapt to custom research projects.

## API
-----

### [mindaudio.data](/mindaudio/data)

- io
- features
- datasets
- augment
- collate
- masks

### [mindaudio.models](/mindaudio/models)
- conformer
- deepspeech2
- tacotron2
- more coming soon...

### [mindaudio.utils](/mindaudio/utils)
- callback
- initializer
- scheduler
- train_one_step

### [mindaudio.adapter](/mindaudio/adapter)
- local_adapter

## Installation
-----

1. Install dependency
```shell
pip install -r requirements.txt
```

2. Install [MindSpore](https://www.mindspore.cn/install)

3. (optional) and [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) for distributed mode.   

4. Install mindaudio

```shell
git clone https://github.com/mindlab-ai/mindaudio.git
cd mindaudio
python setup.py install
```

## Getting Started
-----

### Tacotron2

- Preprocess LJSpeech

```shell
python examples/tacotron2/preprocess_tacotron2_ljspeech.py --config_path examples/tacotron2/config.yaml
```


- Standalone Training, [config](mindaudio/examples/tacotron2/config.yaml)

```shell
python examples/tacotron2/train.py --config_path examples/tacotron2/config.yaml
```

- Distributed Training 

```
export CUDA_VISIBLE_DEVICES=0,1,2,3  # suppose there are 4 GPUs
mpirun --allow-run-as-root -n 4 python train.py --distribute \
	--model=densenet121 --dataset=imagenet --data_dir=./datasets/imagenet   
```

- Inference

To validate the model, you can use `eval.py`. Here is an example.
```shell
python examples/tacotron2/eval.py --model=densenet121 --dataset=imagenet --val_split=val \
		           --ckpt_path='./ckpt/densenet121-best.ckpt' 
``` 

## Notes

### What's New 

- 2022/09/23

Initial version under review

### License

This project is released under the [Apache License 2.0](LICENSE).

### Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [issue](https://github.com/mindlab-ai/mindaudio/issues).

### Acknowledgement

MindSpore is an open source project that welcome any contribution and feedback. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible as well as standardized toolkit to reimplement existing methods and develop their own new Audio methods.

### Contributing

We appreciate all contributions to improve MindSpore Audio. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for the contributing guideline.

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{MindSpore Audio 2022,
    title={{MindSpore Audio}:MindSpore Audio Toolbox and Benchmark},
    author={MindSpore Audio Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindaudio/}},
    year={2022}
}
```