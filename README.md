# MindSpore Audio

## Introduction
MindSpore Audio is an open source audio research toolbox based on MindSpore in audio direction. Mainly focused on rapid research and development on audio tasks, we provide numerous audio processing APIs, deep learning model implementations, as well as example preprocess-train-infer pipeline python scripts for academic purposes. You could easily adapt to your own choice of data processing and model designs for your projects!

### Data API
- stft
- istft
- etc...

### Models
- conformer
- deepspeech2
- tacotron2
- etc...

Under construction... 

## Installation

### Dependency

- mindspore >= 1.8.1
- numpy >= 1.17.0
- pyyaml >= 5.3
- tqdm
- sentencepiece
- openmpi 4.0.3 (for distributed mode) 

To install the dependency, please run
```shell
pip install -r requirements.txt
```

MindSpore can be easily installed by following the official [instruction](https://www.mindspore.cn/install) where you can select your hardware platform for the best fit. To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.   

The following instructions assume the desired dependency is fulfilled. 

TODO

### Install with pip
MindAudio can be installed with pip. 
```shell
pip install https:// xxx .whl
```

### Install from source
To install MindAudio from source, please run,
```shell
# Clone the mindaudio repository.
git clone https://github.com/mindlab-ai/mindaudio.git
cd mindaudio

# Install
python setup.py install
```

## Get Started 

### Hands-on Demo
Please see the [Quick Start Demo](quick_tour.ipynb) to help you get started with MindAudio and learn about the basic usage quickly. 


```python
>>> import mindaudio 
# Search a wanted pretrained model 
>>> mindaudio.list_models("densenet*", pretrain=True)
['densenet201', 'densenet161', 'densenet169', 'densenet121']
# Create the model object
>>> network = mindaudio.create_model('densenet121', pretrained=True)
```

### Quick Running Scripts
It is easy to train your model on standard datasets or your own dataset with MindAudio. Model training, transfer learning, or evaluaiton can be done using one or a few line of code with flexible configuration.

Here's an example on training your own Tacotron2 model on LJSpeech dataset:

- Preprocess LJSpeech

`python examples/tacotron2/preprocess_tacotron2_ljspeech.py --config_path examples/tacotron2/config.yaml`

- Standalone Training

`python examples/tacotron2/train.py --config_path examples/tacotron2/config.yaml`

Detailed adjustable parameters and their default value can be seen in [config.yaml](mindaudio/examples/tacotron2/config.yaml)

- Distributed Training 

```
export CUDA_VISIBLE_DEVICES=0,1,2,3  # suppose there are 4 GPUs
mpirun --allow-run-as-root -n 4 python train.py --distribute \
	--model=densenet121 --dataset=imagenet --data_dir=./datasets/imagenet   
```

- Evaluation

To validate the model, you can use `validate.py`. Here is an example.
```shell
python validate.py --model=densenet121 --dataset=imagenet --val_split=val \
		           --ckpt_path='./ckpt/densenet121-best.ckpt' 
``` 


## Tutorials
We provide [jupyter notebook tutorials](tutorials) for  

- [Learn about configs](tutorials/learn_about_config.ipynb)  //tbc
- [Inference with a pretrained model](tutorials/inference.ipynb) //tbc
- [Finetune a pretrained model on custom datasets](tutorials/finetune.ipynb) 
- [Customize models](tutorials/customize_model.ipynb) //tbc


## Notes
### What's New 

- 2022/09/23

Initial version under review

### License

This project is released under the [Apache License 2.0](LICENSE.md).

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