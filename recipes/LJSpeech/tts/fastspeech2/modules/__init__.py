import mindspore as ms

if ms.__version__ in {'1.8.1', '1.7.0'}:
    from modules.fastspeech2_v181 import FastSpeech2, FastSpeech2WithLoss
elif ms.__version__ in {'1.9.0'}:
    from modules.fastspeech2_v190 import FastSpeech2, FastSpeech2WithLoss
else:
    raise NotImplementedError('unknown mindspore version: %s' % ms.__version__)
