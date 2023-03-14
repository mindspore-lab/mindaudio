import mindspore.dataset.engine as de
import mindaudio
from mindaudio.utils.distributed import DistributedSampler


class AudioTextBaseDataset():
    def __int__(self, manifest_path="", labels=None):
        self.bins = []
        self.manifest_path = manifest_path
        self.labels = {labels[i]: i for i in range(len(labels))}
        self.collect_data()

    def collect_data(self):
        with open(self.manifest_path) as f:
            json_file = json.load(f)
        root_path = json_file.get("root_path")
        wav_txts = json_file.get("samples")

        for data in wav_txts:
            audio_path = os.path.join(root_path, data["wav_path"])
            audio, _ = mindaudio.read(str(audio_path))
            transcript = data["transcript"]
            transcript =list(filter(None, [self.labels.get(x) for x in list(transcript)]))
            self.bins.append([audio, transcript])

    def __getitem__(self, index):
        audio, trans = self.bins[index]
        return audio, trans

    def __len__(self):
        return len(self.bins)


def create_base_dataset(manifest_path, labels, rank=0, group_size=1):
    """

    Args:
        manifest_path(str): the path of json file
        labels(list): To translate text
        rank(int): For distributed computation
        group_size(int): For distributed computation

    Returns:
        Base dataset contained the path of audio and text

    """
    input_columns = ["audio", "text"]
    dataset = AudioTextBaseDataset(manifest_path, labels)
    sampler = DistributedSampler(dataset, rank, group_size, shuffle=True)
    ds = de.GeneratorDataset(
        dataset,
        column_names=input_columns,
        sampler=sampler)
    return ds


def pad_txt_wav_train(audio, text, BatchInfo):
    batch_script = text
    batch_spect = audio
    freq_size = np.shape(batch_spect[-1])[0]
    batch_size = len(batch_script)
    target_indices = []
    input_length = np.zeros(batch_size, np.float32)
    inputs = np.zeros((batch_size, 1, freq_size, TRAIN_INPUT_PAD_LENGTH), dtype=np.float32)
    # The target length is fixed to this value because Mindspore does not support dynamic shape currently
    # 350 may be greater than the max length of labels in train dataset(LibriSpeech).
    targets = np.ones((batch_size, TRAIN_LABEL_PAD_LENGTH), dtype=np.int32) * BLANK_ID

    for k, spect_, scripts_ in zip(range(batch_size), batch_spect, batch_script):
        seq_length = np.shape(spect_)[1]
        input_length[k] = seq_length
        targets[k, :len(scripts_)] = scripts_
        for m in range(TRAIN_LABEL_PAD_LENGTH):
            target_indices.append([k, m])

        if seq_length <= TRAIN_INPUT_PAD_LENGTH:
            input_length[k] = seq_length
            inputs[k, 0, :, 0:seq_length] = spect_[:, :seq_length]
        else:
            maxstart = seq_length - TRAIN_INPUT_PAD_LENGTH
            start = np.random.randint(maxstart)
            input_length[k] = TRAIN_INPUT_PAD_LENGTH
            inputs[k, 0, :, 0:TRAIN_INPUT_PAD_LENGTH] = spect_[:, start:start + TRAIN_INPUT_PAD_LENGTH]
    targets = np.reshape(targets, (-1,))
    outputs = (inputs, input_length, np.array(target_indices , dtype=np.int64), np.array(targets, dtype=np.int32))
    return outputs


def pad_txt_wav_eval(audio, text, BatchInfo):
    batch_script = text
    batch_spect = audio
    freq_size = np.shape(batch_spect[-1])[0]
    batch_size = len(batch_script)
    target_indices = []
    input_length = np.zeros(batch_size, np.float32)
    inputs = np.zeros((batch_size, 1, freq_size, TEST_INPUT_PAD_LENGTH), dtype=np.float32)
    targets = []
    for k, spect_, scripts_ in zip(range(batch_size), batch_spect, batch_script):
        seq_length = np.shape(spect_)[1]
        input_length[k] = seq_length
        targets.extend(scripts_)
        for m in range(len(scripts_)):
            target_indices.append([k, m])
        inputs[k, 0, :, 0:seq_length] = spect_
    outputs = (inputs, input_length, np.array(target_indices, dtype=np.int64), np.array(targets, dtype=np.int32))
    return outputs


def get_feature(audio, sample_rate, window_size, window_stride):
    n_fft = int(sample_rate * window_size)
    win_length = n_fft
    hop_length = int(sample_rate * window_stride)
    D = mindaudio.stft(waveforms=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag, _ = mindaudio.magphase(D, power=1.0, iscomplex=True)
    mag = np.log1p(mag)
    return mag


def train_data_pipeline(dataset, batch_size, audio_conf):
    ds = dataset.map(lambda x: get_feature(x, audio_conf.sample_rate, audio_conf.window_size, audio_conf.window_stride),
                     input_columns=["audio"],
                     num_parallel_wokers=cpu_count())

    ds = ds.batch(batch_size,
                  per_batch_map=pad_txt_wav_train,
                  input_columns=["audio", "text"],
                  output_columns=["inputs", "input_length", "target_indices", "label_values"])
    return ds


def eval_data_pipeline(dataset, batch_size, audio_conf):
    ds = dataset.map(lambda x: get_feature(x, audio_conf.sample_rate, audio_conf.window_size, audio_conf.window_stride),
                     input_columns=["audio"],
                     num_parallel_wokers=cpu_count())

    ds = ds.batch(batch_size,
                  per_batch_map=pad_txt_wav_eval,
                  input_columns=["audio", "text"],
                  output_columns=["inputs", "input_length", "target_indices", "label_values"])
    return ds