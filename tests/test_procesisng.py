import numpy as np
import os
import sys
sys.path.append('.')
import mindaudio.data.io as io
import mindaudio.data.processing as processing
import mindaudio.data.spectrum as spectrum


def test_normalize():
    waveforms = np.vander(np.arange(-2, 2))
    # Max (L-Infinity)-normalize the rows
    processing.normalize(waveforms, axis=1)


def test_unitarize():
    root_path = os.sys.path[0]
    data_path = os.path.join(root_path, 'samples', 'ASR', 'BAC009S0002W0122.wav')
    waveform, sr = io.read(data_path)
    waveforms = processing.unitarize(waveform)
    print(waveforms)


def test_resample():
    waveform = np.random.random([1, 441000])
    y_8k = processing.resample(waveform, orig_freq=44100, new_freq=16000)
    print(waveform.shape)
    print(y_8k.shape)


def test_rescale():
    root_path = sys.path[0]
    data_path = os.path.join(root_path, 'samples', 'ASR', 'BAC009S0002W0122.wav')
    waveform, sr = io.read(data_path)
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



def test_invert_channels():
    waveform = np.array([1, 2, 3])
    out_waveform = processing.invert_channels(waveform)
    print(out_waveform)

    waveform1 = np.array([[1, 2, 3], [2, 3, 4]])
    out_waveform = processing.invert_channels(waveform1)
    print(out_waveform)


def test_loop():
    times = 3

    waveform = np.array([1, 2, 3])
    out_waveform = processing.loop(waveform, times)
    print(out_waveform)

    waveform = np.array([[1, 2, 3], [2, 3, 4]])
    out_waveform = processing.loop(waveform, times)
    print(out_waveform)


def test_clip():
    offset_factor = 0.1
    duration_factor = 0.3

    waveform = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    out_waveform = processing.clip(waveform, offset_factor, duration_factor)
    print(out_waveform)

    waveform = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]).T
    out_waveform = processing.clip(waveform, offset_factor, duration_factor)
    print(out_waveform)


def test_insert_in_background():
    offset_factor = 0.2
    waveform1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    background_audio1 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    out_waveform1 = processing.insert_in_background(waveform1, offset_factor, background_audio1)
    print(out_waveform1)

    waveform2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    background_audio2 = np.array([[0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]]).T
    out_waveform2 = processing.insert_in_background(waveform2, offset_factor, background_audio2)
    print(out_waveform2)

    waveform3 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]).T
    background_audio3 = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    out_waveform3 = processing.insert_in_background(waveform3, offset_factor, background_audio3)
    print(out_waveform3)

    waveform4 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]).T
    background_audio4 = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]]).T
    out_waveform4 = processing.insert_in_background(waveform4, offset_factor, background_audio4)
    print(out_waveform4)


if __name__ == "__main__":
    test_normalize()
    test_unitarize()
    test_resample()
    test_rescale()
    test_stereo_to_mono()
    test_trim()
    test_split()
    test_sliding_window_cmn()
    test_invert_channels()
    test_loop()
    test_clip()
    test_insert_in_background()
