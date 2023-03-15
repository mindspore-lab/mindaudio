import argparse
import os
import tarfile
from pathlib import Path
import wget
import shutil
import json
import numpy as np


LIBRI_SPEECH_URLS = {
    "train": ["http://www.openslr.org/resources/12/train-clean-100.tar.gz",
              "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
              "http://www.openslr.org/resources/12/train-other-500.tar.gz"],

    "val": ["http://www.openslr.org/resources/12/dev-clean.tar.gz",
            "http://www.openslr.org/resources/12/dev-other.tar.gz"],

    "test_clean": ["http://www.openslr.org/resources/12/test-clean.tar.gz"],
    "test_other": ["http://www.openslr.org/resources/12/test-other.tar.gz"]
}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare Librispeech")
    parser.add_argument("--root_path", type=str,
                        default="", help="The path to store data")
    arg = parser.parse_args()
    prepare_librispeech(arg.root_path, False)





