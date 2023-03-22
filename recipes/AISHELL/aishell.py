import os
import csv
import numpy as np
from pathlib import Path


TAB = '\t'
TXT = '.txt'
WAV = '.wav'


class AISHELL():
    def __init__(
        self,
        data_path='/data/data_aishell',
        manifest_path=None,
        split='train'
    ):
        self.bins = []
        self.manifest_path = manifest_path
        self.path = data_path
        self.pattern = '/wavs/*'
        self.wav_format = WAV
        assert split in {'train', 'dev', 'test'}
        self.split = split
        self.maybe_create_manifest()
        self.collect_data()
        super().__init__()

    def maybe_create_manifest(self):
        if os.path.exists(self.manifest_path):
            print('[manifest] found at', self.manifest_path)
            return

        if not os.path.exists(self.path):
            print('[manifest] data path not found at', self.path)
            return

        transcript_file = os.path.join(self.path, 'transcript', 'aishell_transcript_v0.8.txt')
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_lines = f.readlines()

        audio_transcript = {}
        for line in transcript_lines:
            line = line.strip()
            audio_id, transcript = line.split(' ', 1)
            speaker = audio_id[6: 6 + 4]
            audio_file = f'{audio_id}.wav'
            transcript_path = os.path.join(self.path, 'transcript', f"{audio_id}.txt")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            audio_transcript[audio_id] = transcript_path

        with open(self.manifest_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=TAB, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for dataset in ["train", "dev", "test"]:
                audio_dir = os.path.join(self.path, 'wav', dataset)
                for audio_file in Path(audio_dir).glob('*/*.wav'):
                    audio_file = str(audio_file)
                    speaker = audio_file[-14:-9]
                    audio_id = audio_file[-20:-4]
                    if audio_id not in audio_transcript:
                        print('transcript not found:', audio_id, audio_file)
                        continue
                    transcript_path = audio_transcript[audio_id]
                    csvwriter.writerow((dataset, audio_file, transcript_path))

        return True

    def collect_data(self):
        if not os.path.exists(self.manifest_path):
            return
        with open(self.manifest_path) as file:
            for line in file.readlines():
                dataset, audio_path, trans_path = line.strip().split(TAB)
                if dataset == self.split:
                    self.bins.append([audio_path, trans_path])
        np.random.seed(0)
        np.random.shuffle(self.bins)
        print(f'[aishell] one of {self.split}: {self.bins[0]}')

    def __getitem__(self, index):
        audio_path, trans_path = self.bins[index]
        return audio_path, trans_path

    def __len__(self):
        return len(self.bins)
