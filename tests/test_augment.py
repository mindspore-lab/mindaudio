import numpy as np
import mindaudio.data.io as io
import mindaudio.data.spectrum as spectrum
import mindaudio.data.augment as augment


def test_frequencymasking():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    orignal = spectrum.spectrogram(waveform)
    masked = augment.frequencymasking(orignal, frequency_mask_param=80)
    print(masked.shape)


def test_timemasking():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    orignal = spectrum.spectrogram(waveform)
    masked = augment.timemasking(orignal, frequency_mask_param=80)
    print(masked.shape)


def test_reverberate():
    read_wav_dir = './samples/ASR/BAC009S0002W0122.wav'
    samples, _ = io.read(read_wav_dir)
    print(samples.shape)
    rirs, _ = io.read('./samples/ASR/1089-134686-0001.wav')
    addnoise = augment.reverberate(samples, rirs)
    print(addnoise.shape)


def test_1d():
    read_wav_dir = './samples/ASR/BAC009S0002W0122.wav'
    samples, _ = io.read(read_wav_dir)

    background_list = ['./samples/ASR/1089-134686-0000.wav',
                       './samples/ASR/1089-134686-0001.wav']
    addnoise = augment.add_noise(samples, background_list, 3, 30, 1.0)
    io.write('./result/1D/addnoise_1d.wav', addnoise, 16000)

    rir_list = ['./samples/rir/air_binaural_aula_carolina_0_1_1_90_3_16k.wav',
                './samples/rir/air_binaural_aula_carolina_0_1_2_90_3_16k.wav',
                './samples/rir/air_binaural_aula_carolina_0_1_3_0_3_16k.wav']
    addrir = augment.add_reverb(samples, rir_list, 1.0)
    io.write('./result/1D/addrir_1d.wav', addrir, 16000)


def test_2d():
    read_wav_dir = './samples/ASR/BAC009S0002W0122.wav'
    data1, _ = io.read(read_wav_dir)
    read_wav_di2 = './samples/ASR/BAC009S0002W0124.wav'
    data2, _ = io.read(read_wav_di2)
    shortlen = min(data1.shape[0], data2.shape[0])
    samples = np.append(data1[:shortlen], data2[:shortlen]).reshape(2, shortlen)

    background_list = ['./samples/ASR/1089-134686-0000.wav',
                       './samples/ASR/1089-134686-0001.wav']
    addnoise = augment.add_noise(samples, background_list, 3, 30, 1.0)
    io.write('./result/2D/addnoise.wav', addnoise[0], 16000)

    rir_list = ['./samples/rir/air_binaural_aula_carolina_0_1_1_90_3_16k.wav',
                './samples/rir/air_binaural_aula_carolina_0_1_2_90_3_16k.wav',
                './samples/rir/air_binaural_aula_carolina_0_1_3_0_3_16k.wav']
    addrir = augment.add_reverb(samples, rir_list, 1.0)
    io.write('./result/2D/addrir.wav', addrir[0], 16000)


def test_3d():
    samples = np.random.rand(10, 1, 200960) - 0.5
    background_list = ['./samples/ASR/1089-134686-0000.wav',
                       './samples/ASR/1089-134686-0001.wav']
    addnoise = augment.add_noise(samples, background_list, 3, 30, 1.0)
    print(addnoise.shape)

    rir_list = ['./samples/rir/air_binaural_aula_carolina_0_1_1_90_3_16k.wav',
                './samples/rir/air_binaural_aula_carolina_0_1_2_90_3_16k.wav',
                './samples/rir/air_binaural_aula_carolina_0_1_3_0_3_16k.wav']
    addrir = augment.add_reverb(samples, rir_list, 1.0)
    print(addrir.shape)


if __name__ == "__main__":
    test_frequencymasking()
    test_timemasking()
    test_reverberate()
    test_1d()
    test_2d()
    test_3d()