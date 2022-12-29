import numpy as np
import mindaudio.data.io as io
import mindaudio.data.processing as processing
import mindaudio.data.spectrum as spectrum


def test_normalize():
    waveforms = np.vander(np.arange(-2, 2))
    # Max (L-Infinity)-normalize the rows
    processing.normalize(waveforms, axis=1)

def test_unitarize():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    waveforms = processing.unitarize(waveform)
    print(waveforms)

def test_resample():
    waveform = np.random.random([1, 441000])
    y_8k = processing.resample(waveform, orig_freq=44100, new_freq=16000)
    print(waveform.shape)
    print(y_8k.shape)

def test_rescale():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    ori_apm = spectrum.compute_amplitude(waveform)
    print(ori_apm)
    target_lvl = 2
    rescaled_waves = processing.rescale(waveform, target_lvl=target_lvl, amp_type="avg")
    apm = spectrum.compute_amplitude(rescaled_waves)
    print(apm)

def test_stereo_to_mono():
    y = np.array([[1, 2], [0.5, 0.1]])
    y = processing.stereo_to_mono(y)
    print(np.allclose(np.array([0.75, 1.05]), y))

def test_trim():
    waveforms = np.array([0.01] * 1000 + [0.6] * 1000 + [-0.6] * 1000)
    wav_trimmed, index = processing.trim(waveforms, top_db=10)
    print(wav_trimmed.shape)

def test_split():
    waveforms = np.array([0.01] * 2048 + [0.6] * 2048 + [-0.01] * 2048 + [0.5] * 2048)
    indices = processing.split(waveforms, top_db=10)
    print(indices.shape)

def test_sliding_window_cmn():
    waveform = np.random.random([1, 20, 10])
    after_CMN = processing.sliding_window_cmn(waveform, 500, 200)
    print(after_CMN)



if __name__ == "__main__":
    test_normalize()
    test_unitarize()
    test_resample()
    test_rescale()
    test_stereo_to_mono()
    test_trim()
    test_split()
    test_sliding_window_cmn()