# A simple parser for librispeech dataset directory structure

import os
import sys
from pathlib import Path
import csv

sys.path.append('..')
from mindaudio.data.datasets import AudioTextBaseDataset, TAB


TXT = '.txt'
FLAC = '.flac'
WAV = '.wav'


class LibriSpeechASR(AudioTextBaseDataset):
    def __init__(
        self,                 
        data_path='/data/LibriSpeech/dev-clean',
        manifest_path=None
    ):
        self.path = data_path
        self.pattern = '*/*/*'
        self.txt_format = '.trans' + TXT
        self.wav_format = FLAC
        super(LibriSpeechASR, self).__init__(manifest_path=manifest_path)

    def maybe_create_manifest(self):
        if os.path.exists(self.manifest_path):
            print('[manifest] found at', self.manifest_path)
            return

        data = []
        root = self.path
        all_files = list(map(str, Path(root).glob(self.pattern + self.txt_format)))
        print(f"[manifest] Searching {root} ... total files:", len(all_files))
        assert len(all_files) > 0, "no files found here!"
        for filename in all_files:
            with open(filename) as f:
                for line in f.readlines():
                    try:
                        book, text = line.strip().split(' ', 1)
                    except:
                        print('broken line:')
                        print(line)
                        exit()
                    text_path = os.path.join(filename[:filename.rfind('/')], book + TXT)
                    wav_path = text_path.replace(TXT, self.wav_format)
                    with open(text_path, 'w') as f2:
                        f2.write(text + '\n')
                    data.append((wav_path, text_path))

        with open(self.manifest_path, 'w') as file:
            writer = csv.writer(file, delimiter=TAB)
            for line in data:
                writer.writerow(line)

        return True
