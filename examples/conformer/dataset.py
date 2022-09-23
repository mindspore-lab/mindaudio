"""
Create train or eval dataset for conformer.
"""
import sys
import numpy as np
from multiprocessing import cpu_count

sys.path.append('..')
from mindaudio.data.io import read
from mindaudio.data.datasets import LibrittsASR, create_asr_dataset
from mindaudio.data.features import Fbank, PreEmphasis
from mindaudio.data.augment import speed_perturb, spec_aug
from mindaudio.data.masks import make_pad_mask, subsequent_mask
from mindaudio.data.collate import pad_batch_text_tokens_train, pad_sequence
from examples.tokenisers import SubwordTokeniser


def create_dataset(
    data_path,
    manifest_path,
    dataset_conf,
    collate_conf,
    rank=0,
    group_size=1,
):
    # raw dataset, each item: [path_to_wav, path_to_text]

    ds = LibrittsASR(
        data_path=data_path,
        manifest_path=manifest_path,
    )
    ds = create_asr_dataset(ds, rank, group_size)

    # process audio: file -> wav -> feature
    
    def read_wav(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        audio, _ = read(filename)
        audio = (audio + 32768) / (32767 + 32768) * 2 - 1
        return audio
    operations = [read_wav]

    if collate_conf['speeds'] is not None:
        def speed_perturb_(x):
            return speed_perturb(x, dataset_conf['sample_rate'], collate_conf['speeds'])
        operations.append(speed_perturb_)

    preemphasis = PreEmphasis()
    operations.append(lambda x: preemphasis(x))

    fbank = Fbank(
        deltas=False,
        context=False,
        n_mels=collate_conf['feature_extraction_conf']['mel_bins'],
        n_fft=512,
        sample_rate=dataset_conf['sample_rate'],
        f_min=0.0,
        f_max=None,
        left_frames=5,
        right_frames=5,
        win_length=int(dataset_conf['sample_rate'] * collate_conf['feature_extraction_conf']['frame_length'] / 1000), 
        hop_length=int(dataset_conf['sample_rate'] * collate_conf['feature_extraction_conf']['frame_shift'] / 1000), 
        window="hann"
    )
    operations.append(lambda x: fbank(x).T)

    if collate_conf['spec_aug_conf'] is not None:
        operations.append(lambda x: spec_aug(x, collate_conf['spec_aug_conf']))

    ds = ds.map(
        input_columns=['audio'],
        operations=operations,
        num_parallel_workers=cpu_count(),
    )

    # process text: file -> text -> token ids

    tokeniser = SubwordTokeniser()
    def read_text(filename):
        filename = str(filename).replace('b\'', '').replace('\'', '')
        text = open(filename, encoding='utf8').readline().replace('\n', '')
        ids = tokeniser.tokenise(text)
        return ids

    ds = ds.map(
        input_columns=['text'],
        operations=read_text,
        num_parallel_workers=cpu_count(),
    )

    # pad xs and ys to same length respectively, then make a batch
        
    sos = tokeniser.sos
    eos = tokeniser.eos
    IGNORE_ID = tokeniser.IGNORE_ID
    def batch_collate(audio, text, unused_batch_info):
        xs, ys = [], []
        max_src_len, max_tgt_len = dataset_conf['min_length'], dataset_conf['token_min_length']
        for x, y in zip(audio, text):
            xs.append(x)
            ys.append(y)
            max_src_len = max(max_src_len, x.shape[0])
            max_tgt_len = max(max_tgt_len, y.shape[0])

        xs_pad = pad_sequence(
            xs,
            padding_value=0.0,
            padding_max_len=max_src_len,
            atype=np.float32,
        )
        ys_pad = pad_sequence(
            ys,
            padding_value=IGNORE_ID,
            padding_max_len=max_tgt_len,
            atype=np.int64,
        )

        # generate the input and output sequence for ASR decoder
        ys_in = [np.concatenate(([sos], y), axis=0) for y in ys]
        ys_out = [np.concatenate((y, [eos]), axis=0) for y in ys]
        ys_in_pad = pad_sequence(
            ys_in,
            padding_value=eos,
            padding_max_len=max_tgt_len + 1,
            atype=np.int64,
        )
        ys_out_pad = pad_sequence(
            ys_out,
            padding_value=IGNORE_ID,
            padding_max_len=max_tgt_len + 1,
            atype=np.int64,
        )

        xs_lengths = np.array([x.shape[0] for x in xs], dtype=np.int32)
        ys_lengths = np.array([len(y) for y in ys], dtype=np.int32)

        # make xs_masks, (B, 1, T), audio == 1, padding == 0
        xs_masks = np.expand_dims(~make_pad_mask(xs_lengths, max_len=max_src_len), 1)
        xs_masks = xs_masks.astype(np.float32)

        # make ys_masks, (B, 1, T), text == 1, padding == 0
        # the length of each y should be increase by 1 (+ sos / eos)
        ys_masks = np.expand_dims(~make_pad_mask(ys_lengths + 1, max_len=max_tgt_len + 1), 1)
        m = np.expand_dims(subsequent_mask(max_tgt_len + 1), 0)
        ys_sub_masks = (ys_masks & m).astype(np.float32)
        ys_masks = ys_masks.astype(np.float32)
        xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]

        ys_pad_indices, targets = pad_batch_text_tokens_train(ys_pad)

        return (
            xs_pad,
            ys_pad,
            ys_in_pad,
            ys_out_pad,
            xs_masks,
            ys_sub_masks,
            ys_masks,
            ys_pad_indices,
        )

    output_feat = [
        "xs_pad",
        "ys_pad",
        "ys_in_pad",
        "ys_out_pad",
        "xs_masks",
        "ys_masks",
        "ys_sub_masks",
        "ys_lengths",
    ]
    ds = ds.batch(
        collate_conf['batch_size'],
        per_batch_map=batch_collate,
        input_columns=['audio', 'text'],
        output_columns=output_feat,
        column_order=output_feat,
    )

    return tokeniser.vocab_size, ds


def load_language_dict(dict_file):
    """Load dict for ASR."""
    char_dict = {}
    with open(dict_file, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    sos = len(char_dict) - 1
    eos = len(char_dict) - 1
    vocab_size = len(char_dict)
    return sos, eos, vocab_size, char_dict


def create_asr_predict_dataset(data_file, dataset_conf, collate_conf):
    raise NotImplementedError
