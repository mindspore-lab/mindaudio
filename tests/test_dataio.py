import os
import numpy as np
import scipy.io
from scipy.io import wavfile


def test_read_2chanel():
    from os.path import dirname, join as pjoin
    from mindaudio.data.io import read

    #Get a multi-channel audio file from the tests/data directory.
    data_dir = os.path.join(os.path.dirname(scipy.io.__file__), 'tests', 'data')
    wav_fname = os.path.join(data_dir, 'test-44100Hz-2ch-32bit-float-be.wav')

    #Load the .wav file contents.
    audio, sr = read(wav_fname)
    print(f"number of channels = {audio.shape[1]}")
    #number of channels = 2
    length = audio.shape[0] / sr
    print(f"length = {length}s")
    #length = 0.01s

    #Plot the waveform.
    import matplotlib.pyplot as plt
    import numpy as np
    time = np.linspace(0., length, audio.shape[0])
    plt.plot(time, audio[:, 0], label="Left channel")
    plt.plot(time, audio[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


def test_read_write():
    from mindaudio.data.io import read, write
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


if __name__ == "__main__":
    test_read_2chanel()
    test_read_write()