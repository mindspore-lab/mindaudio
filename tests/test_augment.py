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


def test_add_babble():
    wav_list = ['./samples/ASR/BAC009S0002W0122.wav',
                './samples/ASR/BAC009S0002W0123.wav',
                './samples/ASR/BAC009S0002W0124.wav',]
    wav_num = 0
    maxlen = 0
    lenlist = []
    for wavdir in wav_list:
        wav, _ = io.read(wavdir)
        wavlen = len(wav)
        lenlist.append(wavlen)
        maxlen = max(wavlen, maxlen)
        if wav_num == 0:
            waveforms = np.expand_dims(wav, axis=0)
        else:
            wav = np.expand_dims(np.pad(wav, (0, maxlen-wavlen), 'constant'), axis=0)
            waveforms = np.concatenate((waveforms, wav), axis=0)
        wav_num += 1
    lengths = np.array(lenlist)/maxlen
    noisy_mindaudio = augment.add_babble(waveforms, lengths, speaker_count=3, snr_low=0, snr_high=0, mix_prob=1.0)


def test_drop_freq():
    signal, _ = io.read('./samples/ASR/1089-134686-0000.wav')
    dropped_signal_mindaudio = augment.drop_freq(signal)


def test_speed_perturb():
    signal, _ = io.read('./samples/ASR/1089-134686-0000.wav')
    perturbed_mindaudio = augment.speed_perturb(signal, orig_freq=16000, speeds=[90])


def test_drop_chunk():
    wav_list = ['./samples/ASR/BAC009S0002W0122.wav',
                './samples/ASR/BAC009S0002W0123.wav',
                './samples/ASR/BAC009S0002W0124.wav', ]
    wav_num = 0
    maxlen = 0
    lenlist = []
    for wavdir in wav_list:
        wav, _ = io.read(wavdir)
        wavlen = len(wav)
        lenlist.append(wavlen)
        maxlen = max(wavlen, maxlen)
        if wav_num == 0:
            waveforms = np.expand_dims(wav, axis=0)
        else:
            wav = np.expand_dims(np.pad(wav, (0, maxlen - wavlen), 'constant'), axis=0)
            waveforms = np.concatenate((waveforms, wav), axis=0)
        wav_num += 1
    lengths = np.array(lenlist) / maxlen
    dropped_waveform = augment.drop_chunk(waveforms, lengths, drop_start=100, drop_end=200, noise_factor=0.0)


if __name__ == "__main__":
    test_frequencymasking()
    test_timemasking()
    test_reverberate()
    test_1d()
    test_2d()
    test_3d()
    test_add_babble()
    test_drop_freq()
    test_speed_perturb()
    test_drop_chunk()