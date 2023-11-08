""" Convert the relevant information in the audio wav file to a json file """

import argparse
import json
import os

import mindaudio


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    """
    sample_rate: 8000
    Read the wav file and save the path and len to the json file
    """
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = mindaudio.read(wav_path)
        # sr=sample_rate
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def preprocess(args):
    """Process all files"""
    for data_type in ["tr", "cv", "tt"]:
        for speaker in ["mix", "s1", "s2"]:
            preprocess_one_dir(
                os.path.join(args.in_dir, data_type, speaker),
                os.path.join(args.out_dir, data_type),
                speaker,
                sample_rate=args.sample_rate,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WSJ0 data preprocessing")
    parser.add_argument(
        "--in-dir",
        type=str,
        default="/dataset/LS-2mix/LS-2mix-data",
        help="Directory path of librimix including tr, cv and tt",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/TasNet/data",
        help="Directory path to put output files",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=8000, help="Sample rate of audio file"
    )
    arg = parser.parse_args()
    print(arg)
    preprocess(arg)
