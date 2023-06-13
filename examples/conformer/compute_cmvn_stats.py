"""compute cmvn stats methods."""

import argparse
import codecs
import csv
import json
import sys

import mindspore.dataset.engine as de
import numpy as np
import yaml
from dataset import compute_fbank_feats

from mindaudio.data.io import read


class AudioDataset:
    """audio dataset definition.

    Args:
        data_file (string): audio data file path
    """

    def __init__(self, data_file, audio_mel_bins, audio_frame_len, audio_frame_shift):
        self.items = []
        # with codecs.open(data_file, "r", encoding="utf-8") as f:
        #     for line in f:
        with open(data_file, "r") as csvfile:
            csvreader = csv.reader(csvfile)
            datanum = 0
            for row in csvreader:
                if datanum > 0:
                    # arr = line.strip().split()
                    print(row)
                    self.items.append(row[2])
                datanum += 1

        self.mel_bins = audio_mel_bins
        self.frame_len = audio_frame_len
        self.frame_shift = audio_frame_shift

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        mean_stat = np.zeros(self.mel_bins)
        var_stat = np.zeros(self.mel_bins)
        number = 0

        waveform, sample_rate = read(self.items[idx])
        waveform = waveform * (1 << 15)

        xs = compute_fbank_feats(
            waveform,
            sample_rate,
            mel_bin=self.mel_bins,
            frame_len=self.frame_len,
            frame_shift=self.frame_shift,
        )

        mean_stat += np.sum(xs, axis=0)
        var_stat += np.sum(np.square(xs), axis=0)
        number += xs.shape[0]
        number = np.array(number)
        return number, mean_stat, var_stat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract CMVN stats")
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="num of subprocess workers for processing",
    )
    parser.add_argument("--train_config", default="", help="training yaml conf")
    parser.add_argument("--in_scp", default=None, help="wav scp file")
    parser.add_argument("--out_cmvn", default="global_cmvn", help="global cmvn file")
    doc = "Print log after every log_interval audios are processed."
    parser.add_argument("--log_interval", type=int, default=1000, help=doc)

    args = parser.parse_args()

    with open(args.train_config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    mel_bins = int(configs["collate_conf"]["feature_extraction_conf"]["mel_bins"])
    frame_len = int(configs["collate_conf"]["feature_extraction_conf"]["frame_length"])
    frame_shift = int(configs["collate_conf"]["feature_extraction_conf"]["frame_shift"])

    resample_rate = 0
    if "resample_conf" in configs["dataset_conf"]:
        resample_rate = configs["dataset_conf"]["resample_conf"]["resample_rate"]
        print("using resample and new sample rate is {}".format(resample_rate))

    batch_size = 1
    dataset = AudioDataset(args.in_scp, mel_bins, frame_len, frame_shift)
    ds = de.GeneratorDataset(
        dataset,
        ["number", "mean_stat", "var_stat"],
        num_parallel_workers=args.num_workers,
        shuffle=False,
    )

    all_number = 0
    all_mean_stat = np.zeros(mel_bins)
    all_var_stat = np.zeros(mel_bins)
    wav_number = 0
    for data in ds.create_dict_iterator(output_numpy=True):
        all_mean_stat += data["mean_stat"]
        all_var_stat += data["var_stat"]
        all_number += data["number"]
        wav_number += batch_size

        if wav_number % args.log_interval == 0:
            print(
                f"processed {wav_number} wavs, {all_number} frames",
                file=sys.stderr,
                flush=True,
            )

    cmvn_info = {
        "mean_stat": list(all_mean_stat.tolist()),
        "var_stat": list(all_var_stat.tolist()),
        "frame_num": all_number.tolist(),
    }

    with open(args.out_cmvn, "w") as fout:
        fout.write(json.dumps(cmvn_info))
