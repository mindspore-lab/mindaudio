import re
import os
import argparse
from string import punctuation

import yaml
import numpy as np

from g2p_en import G2p
from pypinyin import pinyin, Style

from recipes.LJSpeech.text import text_to_sequence

import mindspore as ms
from modules.fastspeech2_v190 import FastSpeech2
from hparams import hps


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, hps):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(hps.lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence:", (phones))
    # a, b = phones.split('OW1 P AH0 N')
    # phones = a + 'OW1 P AH0 N sp' + b
    sequence = np.array(
        text_to_sequence(
            phones, ["english_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, hps):
    lexicon = read_lexicon(hps.lexicon_path)

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, []
        )
    )

    return np.array(sequence)


def synthesize(model, batchs):
    print('batchs:', len(batchs))
    for i, batch in enumerate(batchs):
        output = model(
            p_control=1.,
            e_control=1.,
            d_control=1.,
            **batch
        )
        for k, v in output.items():
            print(k, ':', v.shape)
        mel_len = output['mel_len'].asnumpy()
        mel_prediction = output['mel_predictions'].asnumpy()
        np.save('msfs2_%s.npy' % i, mel_prediction)

def parse_args():
    parser = argparse.ArgumentParser(description='FastSpeech2 training')
    parser.add_argument('--device_target', type=str, default="GPU", choices=("GPU", "CPU", 'Ascend'))
    parser.add_argument('--device_id', '-i', type=int, default=0)
    parser.add_argument('--context_mode', type=str, default='py', choices=['py', 'graph'])
    parser.add_argument('--text', '-t', type=str, default='this is a test text')
    parser.add_argument('--restore', '-r', type=str, default='')
    parser.add_argument('--data_url', default='')
    parser.add_argument('--train_url', default='')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    hps.model.transformer.encoder_dropout = hps.model.transformer.decoder_dropout = 0.

    mode = ms.context.PYNATIVE_MODE if args.context_mode == 'py' else ms.context.GRAPH_MODE
    ms.context.set_context(mode=mode, device_target=args.device_target)
    rank = int(os.getenv('DEVICE_ID', 0))
    group = int(os.getenv('RANK_SIZE', '1'))
    print('[info] rank: %d group: %d' % (rank, group))
    ms.context.set_context(device_id=args.device_id)

    np.random.seed(0)
    ms.set_seed(0)

    model = FastSpeech2(hps)
    if os.path.exists(args.restore):
        print('[info] restore model from', args.restore)
        ms.load_checkpoint(args.restore, model, strict_load=True)

    ids = raw_texts = [args.text[:100]]
    speakers = ms.Tensor([0])
    texts = ms.Tensor([preprocess_english(args.text, hps)])
    text_lens = ms.Tensor([len(texts[0])])
    batches = {
        'speakers': speakers,
        'texts': texts,
        'src_lens': text_lens,
        'max_src_len': ms.Tensor(max(text_lens)),
    }

    synthesize(model, [batches])

if __name__ == "__main__":
    main()
