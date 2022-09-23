# A simple parser for ljspeech dataset directory structure

import os
import sys
import csv

sys.path.append('..')
from mindaudio.data.datasets import AudioTextBaseDataset


TXT = '.txt'
WAV = '.wav'


class LJSpeechTTS(AudioTextBaseDataset):
    def __init__(
        self,
        data_path='/data/LJSpeech-1.1',
        manifest_path=None
    ):
        self.path = data_path
        self.pattern = '/wavs/*'
        self.txt_format = 'metadata.csv'
        self.wav_format = WAV
        super(LJSpeechTTS, self).__init__(manifest_path=manifest_path)

    def maybe_create_manifest(self):
        if os.path.exists(self.manifest_path):
            print('[manifest] found at', self.manifest_path)
            return

        csv_file = os.path.join(self.path, self.txt_format)
        assert os.path.isfile(csv_file), "no files found here: %s" % csv_file

        wav_dir = os.path.join(self.path, 'wavs')
        txt_dir = os.path.join(self.path, 'txts')
        os.makedirs(txt_dir, exist_ok=True)
        
        data = []
        with open(csv_file) as f:
            for line in f.readlines():
                parts = line.strip().split('|')
                name, text = parts[:2]
                text_path = os.path.join(txt_dir, name + TXT)
                wav_path = os.path.join(wav_dir, name + self.wav_format)
                with open(text_path, 'w') as f2:
                    f2.write(text + '\n')
                data.append((wav_path, text_path))

        with open(self.manifest_path, 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            for line in data:
                writer.writerow(line)

        return True
