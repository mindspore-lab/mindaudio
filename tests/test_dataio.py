import os
import numpy as np
import scipy.io
from scipy.io import wavfile


def test_read_write():
    from mindaudio.data.io.dataio import read, write
    data_dir = os.path.join(os.path.dirname(scipy.io.__file__), 'tests', 'data')
    wav_fname = os.path.join(data_dir, 'test-44100Hz-2ch-32bit-float-be.wav')
    samplerate, data = wavfile.read(wav_fname)
    y, sr = read(wav_fname)
    assert isinstance(y, np.ndarray)
    assert np.allclose(sr, samplerate)
    assert np.allclose(data, y)
    assert np.allclose(data.shape[0]/samplerate, y.shape[0]/sr)

    write('test_wav', y, sr)
    y_fromtest, sr_fromtest = read('test_wav')
    assert np.allclose(y_fromtest, y)
    assert sr_fromtest == sr

