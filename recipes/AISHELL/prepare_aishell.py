import os
import shutil
import logging
import wget
import glob
import csv
import numpy as np
import mindspore as ms
import mindaudio.data.io as io

logger = logging.getLogger(__name__)


def download_aishell(data_folder):
    """
    This function prepares the AISHELL-1 dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.
    data_folder : path to AISHELL-1 dataset.
    save_folder: path where to store the manifest csv files.

    """

    # If the data folders do not exist, we need to extract the data
    if not os.path.isdir(os.path.join(data_folder, "data_aishell/wav")):
        # Check for zip file and download if it doesn't exist
        zip_location = os.path.join(data_folder, "data_aishell.tgz")
        if not os.path.exists(zip_location):
            url = "https://www.openslr.org/resources/33/data_aishell.tgz"
            wget.download(url, zip_location)
        logger.info("Extracting data_aishell.tgz...")
        shutil.unpack_archive(zip_location, data_folder)
        wav_dir = os.path.join(data_folder, "data_aishell/wav")
        tgz_list = glob.glob(wav_dir + "/*.tar.gz")
        for tgz in tgz_list:
            shutil.unpack_archive(tgz, wav_dir)
            os.remove(tgz)

def save_infocsv(save_folder):
    # Create filename-to-transcript dictionary
    filename2transcript = {}
    with open(
            os.path.join(
                data_folder, "data_aishell/transcript/aishell_transcript_v0.8.txt"
            ),
            "r",
    ) as f:
        lines = f.readlines()
        for line in lines:
            key = line.split()[0]
            value = " ".join(line.split()[1:])
            filename2transcript[key] = value

    splits = [
        "train",
        "dev",
        "test",
    ]
    ID_start = 0  # needed to have a unique ID for each audio
    for split in splits:
        new_filename = os.path.join(save_folder, split) + ".csv"
        if os.path.exists(new_filename):
            continue
        logger.info("Preparing %s..." % new_filename)

        csv_output = [["ID", "duration", "wav", "transcript"]]
        entry = []

        all_wavs = glob.glob(
            os.path.join(data_folder, "data_aishell/wav")
            + "/"
            + split
            + "/*/*.wav"
        )
        for i in range(len(all_wavs)):
            filename = all_wavs[i].split("/")[-1].split(".wav")[0]
            if filename not in filename2transcript:
                continue
            signal, _ = io.read(all_wavs[i])
            duration = signal.shape[0]
            transcript_ = filename2transcript[filename]
            csv_line = [
                ID_start + i,
                all_wavs[i],
                str(duration),
                transcript_,
            ]
            entry.append(csv_line)

        csv_output = csv_output + entry

        with open(new_filename, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for line in csv_output:
                csv_writer.writerow(line)

        msg = "\t%s successfully created!" % (new_filename)
        logger.info(msg)

        ID_start += len(all_wavs)


class DistributedSampler:
    """Function to distribute and shuffle sample

    Args:
        dataset (BucketAsrDataset): init dataset instance.
        rank (int): current rank of the total world size, for multi-GPUs distributed training
        group_size (int): number of total world size, for multi-GPUs distributed training.
        shuffle (bool): whether to shuffle the dataset.
        seed (int): random seed.
    """
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xFFFFFFFF
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len).tolist()
        else:
            indices = list(range(self.dataset_len))
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.dataset_len


class AudioTextBaseDataset():
    """
    Each item of this dataset consists of a tuple (audio_path, trans_path):
        audio_path: str, path to audio file
        trans_path: str, path to transcript file
    """
    def __init__(self, data_path, manifest_path):
        self.bins = []
        self.data_path = data_path
        self.manifest_path = manifest_path
        self.collect_data()

    def collect_data(self):
        with open(self.manifest_path, 'r') as file:

            reader = csv.reader(file)
            num = 0
            for line in reader:
                if num > 0:

                    audio_path, trans_path = line[1], line[3]
                    self.bins.append([audio_path, trans_path])
                if num > 50:
                    return
                num = num+1

    def __getitem__(self, index):
        audio_path, trans_path = self.bins[index]
        return audio_path, trans_path

    def __len__(self):
        return len(self.bins)


def aishell_Base(rank=None, group_size=None, data_path='', manifest_path=''):
    """
    create base dataset
    Args:
        rank (int): current rank of the total world size, for multi-GPUs distributed training.
        group_size (int): number of total world size, for multi-GPUs distributed training.
    """
    dataset = AudioTextBaseDataset(data_path, manifest_path)
    input_columns = ["audio", "text"]
    sampler = DistributedSampler(dataset, rank, group_size, shuffle=True)
    ds = ms.dataset.GeneratorDataset(
        dataset,
        column_names=input_columns,
        sampler=sampler
    )
    return ds


if __name__ == "__main__":
    data_folder = './data'
    save_folder = './data'





