import argparse
import os
import tarfile
from pathlib import Path
import wget
import shutil
import json
import numpy as np
from  multiprocessing import cpu_count

import mindspore.dataset.engine as de
import mindaudio
from mindaudio.utils.distributed import DistributedSampler



LIBRI_SPEECH_URLS = {
    "train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz",
              "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
              "http://www.openslr.org/resources/12/train-other-500.tar.gz"],

    "val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz",
            "http://www.openslr.org/resources/12/dev-other.tar.gz"],

    "test_clean": ["http://www.openslr.org/resources/12/test-clean.tar.gz"],
    "test_other": ["http://www.openslr.org/resources/12/test-other.tar.gz"]
}


TRAIN_INPUT_PAD_LENGTH = 1600#1250
TRAIN_LABEL_PAD_LENGTH = 500#350
TEST_INPUT_PAD_LENGTH = 4000
BLANK_ID = 28


def _download_data(root):
    for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
        print(split_type, lst_libri_urls)
        for url in lst_libri_urls:
            filename = url.split("/")[-1]
            target_filename = os.path.join(root, filename)
            if not os.path.exists(target_filename):
                wget.download(url, root)


def creat_json_dict(root):
    #root = '/home/litingyu/data/LibriSpeech'
    for dataset_type, libri_urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(root, dataset_type)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        json_file = {
            'root_path': split_dir,
            'samples': []
        }

        wav_dir = os.path.join(split_dir, 'wav')
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)
        for url in libri_urls:
            filename = url.split("/")[-1]
            target_filename = os.path.join(root, filename)
            tar = tarfile.open(target_filename)
            tar.extractall(root)
            tar.close()
            data_path = os.path.join(root, 'LibriSpeech')
            file_paths = list(Path(data_path).rglob(f"*.{'txt'}"))

            for txt_path in file_paths:
                base_path = str(txt_path).split(".")[0]
                transcriptions = open(txt_path).read().strip().split("\n")
                transcriptions = {t.split()[0]: " ".join(t.split()[1:]) for t in transcriptions}
                for item in transcriptions.items():
                    new_wav_path = os.path.join("wav", str(item[0])+".wav")
                    transcript = item[1]
                    json_file['samples'].append({
                        'wav_path': new_wav_path,
                        'transcript': transcript,
                    })
                    wav_path = base_path + "-" + str(item[0].split("-")[-1]) + ".flac"
                    shutil.move(wav_path, wav_dir)
            output_path = Path(os.path.join(split_dir, 'libri_' + dataset_type + '_manifest.json'))
            output_path.write_text(json.dumps(json_file), encoding='utf8')
            shutil.rmtree(data_path)


def prepare_librispeech(root, data_ready):
    if data_ready:
        _download_data(root)
    creat_json_dict(root)


class AudioTextBaseDataset():
    def __int__(self, manifest_path="", labels=None):
        self.bins = []
        self.manifest_path = manifest_path
        self.labels = {labels[i]: i for i in range(len(labels))}
        self.collect_data()

    def collect_data(self):
        with open(self.manifest_path) as f:
            json_file = json.load(f)
        root_path = json_file.get("root_path")
        wav_txts = json_file.get("samples")

        for data in wav_txts:
            audio_path = os.path.join(root_path, data["wav_path"])
            audio, _ = mindaudio.read(str(audio_path))
            transcript =list(filter(None, [self.labels.get(x) for x in list(transcript)]))
            self.bins.apepnd([audio, transcript])

    def __getitem__(self, index):
        audio, trans = self.bins[index]
        return audio, trans

    def __len__(self):
        return len(self.bins)


def create_base_dataset(manifest_path, labels, rank=0, group_size=1):
    """

    Args:
        manifest_path:
        labels:
        rank:
        group_size:

    Returns:

    """
    input_columns = ["audio", "text"]
    dataset = AudioTextBaseDataset()
    sampler = DistributedSampler(dataset, rank, group_size, shuffle=True)
    ds = de.GeneratorDataset(
        dataset,
        column_names=input_columns,
        sampler=sampler)
    return ds


def pad_txt_wav_train(audio, text, BatchInfo):
    batch_script = text
    batch_spect = audio
    freq_size = np.shape(batch_spect)[1][0]
    batch_size = len(batch_script)
    target_indices = []
    input_length = np.zeros(batch_size, np.float32)
    inputs = np.zeros((batch_size, 1, freq_size, TRAIN_INPUT_PAD_LENGTH), dtype=np.float32)
    # The target length is fixed to this value because Mindspore does not support dynamic shape currently
    # 350 may be greater than the max length of labels in train dataset(LibriSpeech).
    targets = np.ones((batch_size, TRAIN_LABEL_PAD_LENGTH), dtype=np.int32) * BLANK_ID

    for k, spect_, scripts_ in zip(range(batch_size), batch_spect, batch_script):
        seq_length = np.shape(spect_)[1]
        input_length[k] = seq_length
        targets[k, :len(scripts_)] = scripts_
        target_indices.append([k, m] for m in range(TRAIN_LABEL_PAD_LENGTH))

        if seq_length <= TRAIN_INPUT_PAD_LENGTH:
            input_length[k] = seq_length
            inputs[k, 0, :, 0:seq_length] = spect_[:, :seq_length]
        else:
            maxstart = seq_length - TRAIN_INPUT_PAD_LENGTH
            start = np.random.randint(maxstart)
            input_length[k] = TRAIN_INPUT_PAD_LENGTH
            inputs[k, 0, :, 0:TRAIN_INPUT_PAD_LENGTH] = spect_[:, start:start + TRAIN_INPUT_PAD_LENGTH]
    targets = np.reshape(targets, (-1,))
    outputs = (inputs, input_length, np.array(target_indices , dtype=np.int64), np.array(targets, dtype=np.int32))
    return outputs


def get_feature(audio, sample_rate, window_size, window_stride):
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    D = mindaudio.stft(waveforms=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag, _ = mindaudio.magphase(D, power=1.0, iscomplex=True)
    mag = np.log1p(mag)
    return mag


def train_data_pipeline(dataset, batch_size, audio_conf):
    ds = dataset.map(lambda x:get_feature(x, audio_conf.sample_rate, audio_conf.window_size, audio_conf.window_stride),
                     input_columns=["audio"],
                     num_parallel_wokers=cpu_count())

    ds = ds.batch(batch_size,
                  per_batch_map=pad_txt_wav_train,
                  input_columns=["audio", "text"],
                  output_columns = ["inputs", "input_length", "target_indices", "label_values"])
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare Librispeech")
    parser.add_argument("--root_path", type=str,
                        default="", help="The path to store data")
    arg = parser.parse_args()
    prepare_librispeech(arg.root_path, False)





