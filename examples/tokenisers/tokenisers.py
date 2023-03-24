import numpy as np
import sentencepiece as spm

from examples.tokenisers.utils import VOCAB_SIZE, text_to_sequence


class CharTokeniser:
    def __init__(self, labels="'.,abcdefghijklmnopqrstuvwxyz _"):
        self.blank_id = int(labels.index("_"))
        self.labels = {labels[i]: i for i in range(len(labels))}
        self.ids = dict(zip(range(len(labels)), labels))

    @property
    def IGNORE_ID(self):
        return -1

    @property
    def sos(self):
        return len(self.labels) - 1

    @property
    def eos(self):
        return len(self.labels) - 1

    def tokenise(self, text):
        text = text.upper().replace("\n", "")
        tokens = filter(
            lambda t: t is not None, [self.labels.get(x) for x in list(text)]
        )
        return np.array(list(tokens), dtype=np.int32)

    def id2txt(self, ids):
        return "".join(
            filter(lambda t: t is not None, [self.ids.get(i) for i in list(ids)])
        )

    @property
    def vocab_size(self):
        return len(self.labels)


class SubwordTokeniser:
    def __init__(self):
        super().__init__()
        self.s = spm.SentencePieceProcessor(model_file="/root/zdy/spm.model")

    @property
    def IGNORE_ID(self):
        return -1

    @property
    def sos(self):
        return self.s.bos_id()

    @property
    def eos(self):
        return self.s.eos_id()

    @property
    def vocab_size(self):
        return self.s.vocab_size()

    def tokenise(self, text):
        return np.array(self.s.EncodeAsIds(text.lower())).astype(np.int32)

    def id2txt(self, ids):
        return self.s.decode_ids(ids)


class ARPAbetTokeniser:
    def __init__(self, text_cleaners=["english_cleaners"]):
        self.text_cleaners = text_cleaners

    def tokenise(self, text):
        return text_to_sequence(text, self.text_cleaners)

    @property
    def vocab_size(self):
        return VOCAB_SIZE
