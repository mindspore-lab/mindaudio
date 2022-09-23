# A simple parser for libritts dataset directory structure

import os
import sys
from pathlib import Path
import csv

sys.path.append('..')
from mindaudio.data.datasets import AudioTextBaseDataset, TAB


TXT = '.txt'
WAV = '.wav'


class LibrittsASR(AudioTextBaseDataset):
    def __init__(
        self,                 
        data_path='/data/LibriTTS/dev-clean',
        manifest_path=None
    ):
        self.path = data_path
        self.pattern = '*/*/*'
        self.txt_format = '.normalized' + TXT
        self.wav_format = WAV
        super(LibrittsASR, self).__init__(manifest_path=manifest_path)

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
            wav_path = filename.replace(self.txt_format, self.wav_format)
            data.append((wav_path, filename))

        with open(self.manifest_path, 'w') as file:
            writer = csv.writer(file, delimiter=TAB)
            for line in data:
                writer.writerow(line)

        return True
