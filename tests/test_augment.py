import numpy as np
import sys
import os

sys.path.append('.')
import mindaudio.data.io as io
import mindaudio.data.spectrum as spectrum
import mindaudio.data.augment as augment


class TestOperators():
    def setup_method(self):
        self.root_path = sys.path[0]
        self.data_path = os.path.join(self.root_path, 'samples', 'ASR', 'BAC009S0002W0122.wav')
        self.test_data, _ = io.read(self.data_path)
        self.rir_list = [os.path.join(self.root_path, 'samples', 'rir', 'air_binaural_aula_carolina_0_1_1_90_3_16k.wav'),
                         os.path.join(self.root_path, 'samples', 'rir', 'air_binaural_aula_carolina_0_1_2_90_3_16k.wav'),
                         os.path.join(self.root_path, 'samples', 'rir', 'air_binaural_aula_carolina_0_1_3_0_3_16k.wav')]
        self.background_list = [os.path.join(self.root_path, 'samples', 'ASR', '1089-134686-0000.wav'),
                                os.path.join(self.root_path, 'samples', 'ASR', '1089-134686-0001.wav')]
        self.wav_list = [os.path.join(self.root_path, 'samples', 'ASR', 'BAC009S0002W0122.wav'),
                         os.path.join(self.root_path, 'samples', 'ASR', 'BAC009S0002W0123.wav'),
                         os.path.join(self.root_path, 'samples', 'ASR', 'BAC009S0002W0124.wav'),]

    def test_frequencymasking(self):
        orignal = spectrum.spectrogram(self.test_data)
        masked = augment.frequencymasking(orignal, frequency_mask_param=80)

    def test_timemasking(self):
        orignal = spectrum.spectrogram(self.test_data)
        masked = augment.timemasking(orignal, frequency_mask_param=80)

    def test_reverberate(self):
        samples, _ = io.read(self.data_path)
        rirs, _ = io.read(self.rir_list[0])
        addnoise = augment.reverberate(samples, rirs)
        print(addnoise.shape)

    def test_1d(self):
        samples, _ = io.read(self.data_path)
        # test add noise for 1d
        addnoise = augment.add_noise(samples, self.background_list, 3, 30, 1.0)
        io.write(os.path.join(self.root_path, 'result', '1D', 'addnoise_1d.wav'), addnoise, 16000)
        # test add reverb for 1d
        addrir = augment.add_reverb(samples, self.rir_list, 1.0)
        io.write(os.path.join(self.root_path, 'result', '1D', 'addrir_1d.wav'), addrir, 16000)


    def test_2d(self):
        data1, _ = io.read(self.wav_list[0])
        data2, _ = io.read(self.wav_list[2])
        shortlen = min(data1.shape[0], data2.shape[0])
        samples = np.append(data1[:shortlen], data2[:shortlen]).reshape(2, shortlen)
        #test add noise for 2d
        addnoise = augment.add_noise(samples, self.background_list, 3, 30, 1.0)
        io.write(os.path.join(self.root_path, 'result', '2D', 'addnoise.wav'), addnoise[0], 16000)
        # test add reverb for 2d
        addrir = augment.add_reverb(samples, self.rir_list, 1.0)
        io.write(os.path.join(self.root_path,'result', '2D', 'addrir.wav'), addrir[0], 16000)


    def test_3d(self):
        samples = np.random.rand(10, 1, 200960) - 0.5
        # test add noise for 3d
        addnoise = augment.add_noise(samples, self.background_list, 3, 30, 1.0)
        print(addnoise.shape)
        # test add reverb for 3d
        addrir = augment.add_reverb(samples, self.rir_list, 1.0)
        print(addrir.shape)


    def test_add_babble(self):
        wav_num = 0
        maxlen = 0
        lenlist = []
        for wavdir in self.wav_list:
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


    def test_drop_freq(self):
        signal, _ = io.read(self.background_list[0])
        dropped_signal_mindaudio = augment.drop_freq(signal)


    def test_speed_perturb(self):
        signal, _ = io.read(self.background_list[0])
        perturbed_mindaudio = augment.speed_perturb(signal, orig_freq=16000, speeds=[90])


    def test_drop_chunk(self):
        wav_num = 0
        maxlen = 0
        lenlist = []
        for wavdir in self.wav_list:
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
    test = TestOperators()
    test.setup_method()
    test.test_frequencymasking()
    test.test_1d()
    test.test_2d()