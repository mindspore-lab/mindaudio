import os
import wget
import tarfile
import subprocess
import json
import sox
import shutil
import math
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np
import mindspore.dataset.engine as de

sys.path.append('.')
import mindaudio.data.io as io


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


def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def _process_file(wav_dir, txt_dir, base_filename, root_dir):
    full_recording_path = os.path.join(root_dir, base_filename)
    assert os.path.exists(full_recording_path) and os.path.exists(root_dir)
    wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(full_recording_path, str(args.sample_rate),
                                                          wav_recording_path)], shell=True)
    # process transcript
    txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
    transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
    assert os.path.exists(transcript_file), "Transcript file {} does not exist.".format(transcript_file)
    transcriptions = open(transcript_file).read().strip().split("\n")
    transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
    with open(txt_transcript_path, "w") as f:
        key = base_filename.replace(".flac", "").split("-")[-1]
        assert key in transcriptions, "{} is not in the transcriptions".format(key)
        f.write(_preprocess_transcript(transcriptions[key]))
        f.flush()


def create_manifest(
        data_path: str,
        output_name: str,
        manifest_path: str,
        num_workers: int,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        file_extension: str = "wav"):
    data_path = os.path.abspath(data_path)
    file_paths = list(Path(data_path).rglob(f"*.{file_extension}"))
    file_paths = order_and_prune_files(
        file_paths=file_paths,
        min_duration=min_duration,
        max_duration=max_duration,
        num_workers=num_workers
    )

    output_path = Path(manifest_path) / output_name
    output_path.parent.mkdir(exist_ok=True, parents=True)

    manifest = {
        'root_path': data_path,
        'samples': []
    }
    for wav_path in tqdm(file_paths, total=len(file_paths)):
        wav_path = wav_path.relative_to(data_path)
        transcript_path = wav_path.parent.with_name("txt") / wav_path.with_suffix(".txt").name
        manifest['samples'].append({
            'wav_path': wav_path.as_posix(),
            'transcript_path': transcript_path.as_posix()
        })

    output_path.write_text(json.dumps(manifest), encoding='utf8')


def _duration_file_path(path):
    return path, sox.file_info.duration(path)


def order_and_prune_files(
        file_paths,
        min_duration,
        max_duration,
        num_workers):
    print("Gathering durations...")
    with Pool(processes=num_workers) as p:
        duration_file_paths = list(tqdm(p.imap(_duration_file_path, file_paths), total=len(file_paths)))
    print("Sorting manifests...")
    if min_duration and max_duration:
        print("Pruning manifests between %d and %d seconds" % (min_duration, max_duration))
        duration_file_paths = [(path, duration) for path, duration in duration_file_paths if
                               min_duration <= duration <= max_duration]

    total_duration = sum([x[1] for x in duration_file_paths])
    print(f"Total duration of split: {total_duration:.4f}s")
    return [x[0] for x in duration_file_paths]  # Remove durations


def prepare_data(args):
    target_dl_dir = args.target_dir
    if not os.path.exists(target_dl_dir):
        os.makedirs(target_dl_dir)
    files_to_dl = args.files_to_use.strip().split(',')
    for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(target_dl_dir, split_type)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        split_wav_dir = os.path.join(split_dir, "wav")
        if not os.path.exists(split_wav_dir):
            os.makedirs(split_wav_dir)
        split_txt_dir = os.path.join(split_dir, "txt")
        if not os.path.exists(split_txt_dir):
            os.makedirs(split_txt_dir)
        extracted_dir = os.path.join(split_dir, "LibriSpeech")
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
        for url in lst_libri_urls:
            # check if we want to dl this file
            dl_flag = False
            for f in files_to_dl:
                if url.find(f) != -1:
                    dl_flag = True
            if not dl_flag:
                print("Skipping url: {}".format(url))
                continue
            filename = url.split("/")[-1]
            target_filename = os.path.join(split_dir, filename)
            if not os.path.exists(target_filename):
                wget.download(url, split_dir)
            print("Unpacking {}...".format(filename))
            tar = tarfile.open(target_filename)
            tar.extractall(split_dir)
            tar.close()
            os.remove(target_filename)
            print("Converting flac files to wav and extracting transcripts...")
            assert os.path.exists(extracted_dir), "Archive {} was not properly uncompressed.".format(filename)
            for root, subdirs, files in tqdm(os.walk(extracted_dir)):
                for f in files:
                    if f.find(".flac") != -1:
                        _process_file(wav_dir=split_wav_dir, txt_dir=split_txt_dir,
                                      base_filename=f, root_dir=root)

            print("Finished {}".format(url))
            shutil.rmtree(extracted_dir)
        if split_type == 'train':  # Prune to min/max duration
            create_manifest(
                data_path=split_dir,
                output_name='libri_' + split_type + '_manifest.json',
                manifest_path=args.manifest_dir,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                num_workers=args.num_workers
            )
        else:
            create_manifest(
                data_path=split_dir,
                output_name='libri_' + split_type + '_manifest.json',
                manifest_path=args.manifest_dir,
                num_workers=args.num_workers
            )


class LoadAudioAndTranscript():
    """
    parse audio and transcript
    """

    def __init__(self,
                 audio_conf=None,
                 normalize=False,
                 labels=None):
        super(LoadAudioAndTranscript, self).__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window
        self.is_normalization = normalize
        self.labels = labels

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r', encoding='utf8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels.get(x) for x in list(transcript)]))
        return transcript


class ASRDataset(LoadAudioAndTranscript):
    """
        create ASRDataset

        Args:
            audio_conf: Config containing the sample rate, window and the window length/stride in seconds
            manifest_filepath (str): manifest_file path.
            labels (list): List containing all the possible characters to map to
            normalize: Apply standard mean and deviation Normalization to audio tensor
            batch_size (int): Dataset batch size (default=32)
        """

    def __init__(self, audio_conf=None,
                 manifest_filepath='',
                 labels=None,
                 normalize=False,
                 batch_size=32,
                 is_training=True):
        with open(manifest_filepath) as f:
            json_file = json.load(f)

        self.root_path = json_file.get('root_path')
        wav_txts = json_file.get('samples')
        ids = [list(x.values()) for x in wav_txts]

        self.is_training = is_training
        self.ids = ids
        self.blank_id = int(labels.index('_'))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        if len(self.ids) % batch_size != 0:
            self.bins = self.bins[:-1]
            self.bins.append(ids[-batch_size:])
        self.size = len(self.bins)
        self.batch_size = batch_size
        self.labels_map = {labels[i]: i for i in range(len(labels))}
        super(ASRDataset, self).__init__(audio_conf, normalize, self.labels_map)

    def __getitem__(self, index):
        batch_idx = self.bins[index]
        batch_size = len(batch_idx)
        batch_spect, batch_script, target_indices = [], [], []
        input_length = np.zeros(batch_size, np.float32)
        for data in batch_idx:
            audio_path, transcript_path = os.path.join(self.root_path, data[0]), os.path.join(self.root_path, data[1])
            #audio_path, transcript_path = data[0], data[1]
            spect = io.read(audio_path)
            transcript = self.parse_transcript(transcript_path)
            batch_spect.append(spect)
            batch_script.append(transcript)
        freq_size = np.shape(batch_spect[-1])[0]

        if self.is_training:
            # 1501 is the max length in train dataset(LibriSpeech).
            # The length is fixed to this value because Mindspore does not support dynamic shape currently
            inputs = np.zeros((batch_size, 1, freq_size, TRAIN_INPUT_PAD_LENGTH), dtype=np.float32)
            # The target length is fixed to this value because Mindspore does not support dynamic shape currently
            # 350 may be greater than the max length of labels in train dataset(LibriSpeech).
            targets = np.ones((self.batch_size, TRAIN_LABEL_PAD_LENGTH), dtype=np.int32) * self.blank_id
            for k, spect_, scripts_ in zip(range(batch_size), batch_spect, batch_script):
                seq_length = np.shape(spect_)[1]

                # input_length[k] = seq_length
                script_length = len(scripts_)
                targets[k, :script_length] = scripts_
                for m in range(TRAIN_LABEL_PAD_LENGTH):
                    target_indices.append([k, m])
                if seq_length <= TRAIN_INPUT_PAD_LENGTH:
                    input_length[k] = seq_length
                    inputs[k, 0, :, 0:seq_length] = spect_[:, :seq_length]
                else:
                    maxstart = seq_length - TRAIN_INPUT_PAD_LENGTH
                    start = np.random.randint(maxstart)
                    input_length[k] = TRAIN_INPUT_PAD_LENGTH
                    inputs[k, 0, :, 0:TRAIN_INPUT_PAD_LENGTH] = spect_[:, start:start + TRAIN_INPUT_PAD_LENGTH]
            targets = np.reshape(targets, (-1,))
        else:
            inputs = np.zeros((batch_size, 1, freq_size, TEST_INPUT_PAD_LENGTH), dtype=np.float32)
            targets = []
            for k, spect_, scripts_ in zip(range(batch_size), batch_spect, batch_script):
                seq_length = np.shape(spect_)[1]
                input_length[k] = seq_length
                targets.extend(scripts_)
                for m in range(len(scripts_)):
                    target_indices.append([k, m])
                inputs[k, 0, :, 0:seq_length] = spect_

        return inputs, input_length, np.array(target_indices, dtype=np.int64), np.array(targets, dtype=np.int32)

    def __len__(self):
        return self.size


class DistributedSampler():
    """
    function to distribute and shuffle sample
    """

    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(self.dataset)
        self.num_samplers = int(math.ceil(self.dataset_len * 1.0 / self.group_size))
        self.total_size = self.num_samplers * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len).tolist()
        else:
            indices = list(range(self.dataset_len))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samplers


def create_dataset(audio_conf, manifest_filepath, labels, normalize, batch_size, train_mode=True,
                   rank=None, group_size=None):
    """
    create train dataset

    Args:
        audio_conf: Config containing the sample rate, window and the window length/stride in seconds
        manifest_filepath (str): manifest_file path.
        labels (list): list containing all the possible characters to map to
        normalize: Apply standard mean and deviation Normalization to audio tensor
        train_mode (bool): Whether dataset is use for train or eval (default=True).
        batch_size (int): Dataset batch size
        rank (int): The shard ID within num_shards (default=None).
        group_size (int): Number of shards that the dataset should be divided into (default=None).

    Returns:
        Dataset.
    """

    dataset = ASRDataset(audio_conf=audio_conf, manifest_filepath=manifest_filepath, labels=labels, normalize=normalize,
                         batch_size=batch_size, is_training=train_mode)

    sampler = DistributedSampler(dataset, rank, group_size, shuffle=True)

    ds = de.GeneratorDataset(dataset, ["inputs", "input_length", "target_indices", "label_values"], sampler=sampler)
    ds = ds.repeat(1)
    return ds


if __name__ == "__main__":
    prepare_data()
