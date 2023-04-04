# ASR with Librispeech
This folder contains data collection and preparation of speech recognition on  the [LibriSpeech](http://www.openslr.org/resources/12).

[LibriSpeech](http://www.openslr.org/resources/12)  is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.

### Dataset Information

- Train Data：
  - train-clean-100: [6.3G] (training set of 100 hours "clean" speech)
  - train-clean-360.tar.gz [23G] (training set of 360 hours "clean" speech)
  - train-other-500.tar.gz [30G] (training set of 500 hours "other" speech)
- Val Data：
  - dev-clean.tar.gz [337M] (development set, "clean" speech)
  - dev-other.tar.gz [314M] (development set, "other", more challenging, speech)
- Test Data:
  - test-clean.tar.gz [346M] (test set, "clean" speech )
  - test-other.tar.gz [328M] (test set, "other" speech )
- Data format：wav and txt files
  - Note：Data will be processed in librispeech.py



### Dataset Preparation

```shell
# Enter the corresponding dataset directory
cd recipes/LibriSpeech
```
If you don't download librispeech recommend using the following methods：

```shell
# Download and creat json
python librispeech_prepare.py --root_path "your_data_path"
```
If you have already downloaded librispeech recommend using the following methods：
```shell
# creat json
python librispeech_prepare.py --root_path "your_data_path"  --data_ready True
```

LibriSpeech stores files with the flac audio format. To use them within MindAudio you have to convert all the flac files into wav files.
You can do the conversion using [ffmpeg](https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830) or [sox](https://sourceforge.net/projects/sox/). This operation might take several minutes.

After the processing, the dataset directory structure is as follows:

```
    ├─ LibriSpeech_dataset
    │  ├── train
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── val
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  ├── test_clean
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
    │  └── test_other
    │  │   ├─libri_test_clean_manifest.json
    │  │   ├─ wav
    │  │   └─ txt
```

The four *.json file stores the absolute path of the corresponding data. In the subsequent model training and verification, please change xx_manifest in the configuration file(xx. yaml) to the corresponding storage address of libri_xx_manifest.json.
