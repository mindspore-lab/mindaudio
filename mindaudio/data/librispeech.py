import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
import wget

LIBRI_SPEECH_URLS = {
    "train": [
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "http://www.openslr.org/resources/12/train-other-500.tar.gz",
    ],
    "val": [
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "http://www.openslr.org/resources/12/dev-other.tar.gz",
    ],
    "test_clean": ["http://www.openslr.org/resources/12/test-clean.tar.gz"],
    "test_other": ["http://www.openslr.org/resources/12/test-other.tar.gz"],
}

__all__ = ["prepare_librispeech"]

def download_data(data_path):
    for split_type, lst_libri_urls in LIBRI_SPEECH_URLS.items():
        print(split_type, lst_libri_urls)
        for url in lst_libri_urls:
            filename = url.split("/")[-1]
            target_filename = os.path.join(data_path, filename)
            if not os.path.exists(target_filename):
                wget.download(url, data_path)


def creat_txt_file(content, root_path, txt_transcript_path):
    txt_path = os.path.join(root_path, txt_transcript_path)
    with open(txt_path, "w") as f:
        f.write(content)
        f.flush()


def create_json_dict(data_path):
    for dataset_type, libri_urls in LIBRI_SPEECH_URLS.items():
        split_dir = os.path.join(data_path, dataset_type)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        json_file = {"data_path": split_dir, "samples": []}

        wav_dir = os.path.join(split_dir, "wav")
        if not os.path.exists(wav_dir):
            os.makedirs(wav_dir)

        txt_dir = os.path.join(split_dir, "txt")
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        for url in libri_urls:
            filename = url.split("/")[-1]
            target_filename = os.path.join(data_path, filename)
            tar = tarfile.open(target_filename)
            tar.extractall(data_path)
            tar.close()
            data_path = os.path.join(data_path, "LibriSpeech")
            file_paths = list(Path(data_path).rglob(f"*.{'txt'}"))

            for txt_path in file_paths:
                base_path = str(txt_path).split(".")[0]
                transcriptions = open(txt_path).read().strip().split("\n")
                transcriptions = {
                    t.split()[0]: " ".join(t.split()[1:]) for t in transcriptions
                }
                for item in transcriptions.items():
                    new_wav_path = os.path.join("wav", str(item[0]) + ".wav")
                    new_txt_path = os.path.join("txt", str(item[0]) + ".txt")
                    transcript = item[1]
                    creat_txt_file(transcript, split_dir, new_txt_path)
                    json_file["samples"].append(
                        {
                            "wav_path": new_wav_path,
                            "txt_path": new_txt_path,
                        }
                    )
                    wav_path = base_path + "-" + str(item[0].split("-")[-1]) + ".flac"
                    shutil.move(wav_path, wav_dir)
            output_path = Path(
                os.path.join(split_dir, "libri_" + dataset_type + "_manifest.json")
            )
            output_path.write_text(json.dumps(json_file), encoding="utf8")
            shutil.rmtree(data_path)


def prepare_librispeech(data_path, download):
    if download:
        download_data(data_path)
    create_json_dict(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare Librispeech")
    parser.add_argument("--data_path", type=str, default="", help="The path to store data")
    parser.add_argument("--download", type=bool, default=False, help="set true to download librispeech datasets")
    arg = parser.parse_args()
    prepare_librispeech(arg.data_path, arg.download)
