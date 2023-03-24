import json
import math
import os
from multiprocessing import cpu_count

import mindaudio
import mindspore.dataset.engine as de
import numpy as np
from mindaudio.utils.distributed import DistributedSampler

TRAIN_INPUT_PAD_LENGTH = 1250
TRAIN_LABEL_PAD_LENGTH = 350
TEST_INPUT_PAD_LENGTH = 3500


class LoadAudioAndTranscript:
    """
    parse audio and transcript
    """

    def __init__(self, audio_conf=None, normalize=False, labels=None):
        super(LoadAudioAndTranscript, self).__init__()
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.is_normalization = normalize
        self.labels = labels

    def parse_audio(self, audio_path):
        """
        parse audio
        """
        audio = mindaudio.read(str(audio_path))
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        spec = mindaudio.stft(
            waveforms=audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        mag, _ = mindaudio.magphase(spec, power=1.0, iscomplex=True)
        mag = np.log1p(mag)
        if self.is_normalization:
            mean = mag.mean()
            std = mag.std()
            mag = (mag - mean) / std
        return mag

    def parse_transcript(self, transcript_path):
        with open(transcript_path, "r", encoding="utf8") as transcript_file:
            transcript = transcript_file.read().replace("\n", "")
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

    def __init__(
        self,
        audio_conf=None,
        manifest_filepath="",
        labels=None,
        normalize=False,
        batch_size=32,
        is_training=True,
    ):
        with open(manifest_filepath) as f:
            json_file = json.load(f)

        self.root_path = json_file.get('root_path')
        wav_txts = json_file.get('samples')
        ids = [list(x.values()) for x in wav_txts]

        self.is_training = is_training
        self.ids = ids
        self.blank_id = int(labels.index("_"))
        self.bins = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
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
            spect = self.parse_audio(audio_path)
            transcript = self.parse_transcript(transcript_path)
            batch_spect.append(spect)
            batch_script.append(transcript)
        freq_size = np.shape(batch_spect[-1])[0]

        if self.is_training:
            # 1501 is the max length in train dataset(LibriSpeech).
            # The length is fixed to this value because Mindspore does not support dynamic shape currently
            inputs = np.zeros(
                (batch_size, 1, freq_size, TRAIN_INPUT_PAD_LENGTH), dtype=np.float32
            )
            # The target length is fixed to this value because Mindspore does not support dynamic shape currently
            # 350 may be greater than the max length of labels in train dataset(LibriSpeech).
            targets = (
                np.ones((self.batch_size, TRAIN_LABEL_PAD_LENGTH), dtype=np.int32)
                * self.blank_id
            )
            for k, spect_, scripts_ in zip(
                range(batch_size), batch_spect, batch_script
            ):
                seq_length = np.shape(spect_)[1]

                # input_length[k] = seq_length
                script_length = len(scripts_)
                targets[k, :script_length] = scripts_
                for m in range(350):
                    target_indices.append([k, m])
                if seq_length <= TRAIN_INPUT_PAD_LENGTH:
                    input_length[k] = seq_length
                    inputs[k, 0, :, 0:seq_length] = spect_[:, :seq_length]
                else:
                    maxstart = seq_length - TRAIN_INPUT_PAD_LENGTH
                    start = np.random.randint(maxstart)
                    input_length[k] = TRAIN_INPUT_PAD_LENGTH
                    inputs[k, 0, :, 0:TRAIN_INPUT_PAD_LENGTH] = spect_[
                        :, start : start + TRAIN_INPUT_PAD_LENGTH
                    ]
            targets = np.reshape(targets, (-1,))
        else:
            inputs = np.zeros(
                (batch_size, 1, freq_size, TEST_INPUT_PAD_LENGTH), dtype=np.float32
            )
            targets = []
            for k, spect_, scripts_ in zip(
                range(batch_size), batch_spect, batch_script
            ):
                seq_length = np.shape(spect_)[1]
                input_length[k] = seq_length
                targets.extend(scripts_)
                for m in range(len(scripts_)):
                    target_indices.append([k, m])
                inputs[k, 0, :, 0:seq_length] = spect_

        return (
            inputs,
            input_length,
            np.array(target_indices, dtype=np.int64),
            np.array(targets, dtype=np.int32),
        )

    def __len__(self):
        return self.size


def create_dataset(
    audio_conf,
    manifest_filepath,
    labels,
    normalize,
    batch_size,
    train_mode=True,
    rank=None,
    group_size=None,
):
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

    dataset = ASRDataset(
        audio_conf=audio_conf,
        manifest_filepath=manifest_filepath,
        labels=labels,
        normalize=normalize,
        batch_size=batch_size,
        is_training=train_mode,
    )

    sampler = DistributedSampler(dataset, rank, group_size, shuffle=True)

    ds = de.GeneratorDataset(dataset, ["inputs", "input_length", "target_indices", "label_values"], sampler=sampler)
    ds = ds.repeat(1)
    return ds
