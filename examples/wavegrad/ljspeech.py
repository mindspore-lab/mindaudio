import csv
import os
import numpy as np
import mindspore as ms

from mindaudio.utils.distributed import DistributedSampler


TAB = "\t"
TXT = ".txt"
WAV = ".wav"


class LJSpeech:
    def __init__(
        self, data_path="/data/LJSpeech-1.1", manifest_path=None, is_train=True
    ):
        self.bins = []
        self.manifest_path = manifest_path
        self.path = data_path
        self.pattern = "/wavs/*"
        self.txt_format = "metadata.csv"
        self.wav_format = WAV
        self.is_train = is_train
        self.maybe_create_manifest()
        self.collect_data()
        super().__init__()

    def maybe_create_manifest(self):
        if os.path.exists(self.manifest_path):
            print("[manifest] found at", self.manifest_path)
            return

        if not os.path.exists(self.path):
            print("[manifest] data path not found at", self.path)
            return

        csv_file = os.path.join(self.path, self.txt_format)
        assert os.path.isfile(csv_file), "no files found here: %s" % csv_file

        wav_dir = os.path.join(self.path, "wavs")
        txt_dir = os.path.join(self.path, "txts")
        os.makedirs(txt_dir, exist_ok=True)

        data = []
        with open(csv_file) as f:
            for line in f.readlines():
                parts = line.strip().split("|")
                name, text = parts[:2]
                text_path = os.path.join(txt_dir, name + TXT)
                wav_path = os.path.join(wav_dir, name + self.wav_format)
                with open(text_path, "w") as f2:
                    f2.write(text + "\n")
                data.append((wav_path, text_path))

        with open(self.manifest_path, "w") as file:
            writer = csv.writer(file, delimiter="\t")
            for line in data:
                writer.writerow(line)

        return True

    def collect_data(self):
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path) as file:
            for line in file.readlines():
                audio_path, trans_path = line.strip().split(TAB)
                self.bins.append([audio_path, trans_path])
        np.random.seed(0)
        np.random.shuffle(self.bins)
        print(
            "[ljspeech] one of train: %s, one of eval:" % self.bins[0][0],
            self.bins[-1][0],
        )
        if self.is_train:  # 12969
            self.bins = self.bins[: int(0.99 * len(self.bins))]
        else:  # 131
            self.bins = self.bins[int(0.99 * len(self.bins)) :]

    def __getitem__(self, index):
        audio_path, trans_path = self.bins[index]
        return audio_path, trans_path

    def __len__(self):
        return len(self.bins)


def create_ljspeech_tts_dataset(
    ds,
    rank: int = 0,
    group_size: int = 1,
):
    input_columns = ["audio", "text"]
    sampler = DistributedSampler(ds, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(ds, column_names=input_columns, sampler=sampler)
    return ds
