import argparse
import csv
import glob
import logging
import os
import shutil
import wget
import mindaudio

logger = logging.getLogger(__name__)

__all__ = ["prepare_aishell"]

def download_aishell(data_folder):
    """
    This function prepares the AISHELL-1 dataset.
    If the folder does not exist, the zip file will be extracted. If the zip file does not exist, it will be downloaded.
    data_folder : path to AISHELL-1 dataset.

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


def save_aishell_info(data_folder, save_folder):
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
            os.path.join(data_folder, "data_aishell/wav") + "/" + split + "/*/*.wav"
        )
        for i in range(len(all_wavs)):
            filename = all_wavs[i].split("/")[-1].split(".wav")[0]
            if filename not in filename2transcript:
                continue
            signal, _ = mindaudio.read(all_wavs[i])
            duration = signal.shape[0] / 16000
            transcript_ = filename2transcript[filename]
            csv_line = [
                ID_start + i,
                str(duration),
                all_wavs[i],
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


def prepare_aishell(data_path, download=False):
    if download:
        download_aishell(data_path)
    save_aishell_info(data_path, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare aishell")
    parser.add_argument("--data_path", type=str, default="", help="The path to store data")
    parser.add_argument("--download", type=bool, default=False, help="set true to download aishell datasets")
    arg = parser.parse_args()
    prepare_aishell(arg.data_path, arg.download)
