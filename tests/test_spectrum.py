import numpy as np
import os
import sys
sys.path.append('.')
import mindaudio.data.io as io
import mindaudio.data.spectrum as spectrum


class TestOperators():

    def setup_method(self):
        self.root_path = sys.path[0]
        self.data_path = os.path.join(self.root_path, 'samples', 'ASR', 'BAC009S0002W0122.wav')
        self.test_data, self.sr = io.read(self.data_path)

    def test_amplitude_to_dB(self):
        waveforms = np.random.random([1, 400 // 2 + 1, 30])
        out = spectrum.amplitude_to_dB(waveforms)
        print(out.shape)

    def test_dB_to_amplitude(self):
        specgram = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
        out = spectrum.dB_to_amplitude(specgram, 0.5, 0.5)
        print(out.shape)

    def test_stft(self):
        matrix = spectrum.stft(self.test_data)
        print(matrix.shape)

    def test_istft(self):
        matrix = spectrum.stft(self.test_data)
        res = spectrum.istft(matrix)
        assert np.allclose(self.test_data[:res.shape[0]], res)

    def test_compute_amplitude(self):
        waveform, sr = io.read(self.data_path)
        amp_avg = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='avg')
        print(amp_avg)
        amp_peak = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='peak')
        print(amp_peak)
        amp_db = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='peak', dB=True)
        print(amp_db)

    def test_spectrogram(self):
        spec = spectrum.spectrogram(self.test_data)
        print(spec.shape)

    def test_melspectrogram(self):
        spec = spectrum.melspectrogram(self.test_data)
        print(spec.shape)

    def test_magphase(self):
        D = spectrum.stft(self.test_data)
        magnitude, phase = spectrum.magphase(D, power=2.0, iscomplex=True)
        print(magnitude, phase)

    def test_melscale(self):
        spec = spectrum.spectrogram(self.test_data, n_fft=1024)
        melscale_spec = spectrum.melscale(spec, n_stft=1024 // 2 + 1)


if __name__ == "__main__":
    test = TestOperators()
    test.setup_method()
    test.test_istft()
    test.test_dB_to_amplitude()