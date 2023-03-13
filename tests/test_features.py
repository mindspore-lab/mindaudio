import numpy as np
import os
import sys
sys.path.append('.')
import mindaudio.data.io as io
import mindaudio.data.features as features
import mindaudio.data.spectrum as spectrum


class TestOperators():

    def setup_method(self):
        self.root_path = sys.path[0]
        self.data_path = os.path.join('samples', 'ASR', 'BAC009S0002W0122.wav')
        self.test_data, self.sr = io.read(os.path.join(self.root_path, self.data_path))

    def test_spectral_centroid(self):
        spectralcentroid = features.spectral_centroid(self.test_data, self.sr)  # (channel, time)
        print(spectralcentroid.shape)

    def test_context_window(self):
        input_arrs = [np.random.randn(10, 101, 60).astype(dtype=np.float32),
                      np.random.randn(10, 101, 60, 2).astype(dtype=np.float32)]
        left_frames = [3, 4, 5, 0]
        right_frames = [5, 4, 3, 0]
        for left, right in zip(left_frames, right_frames):
            for input_arr in input_arrs:
                contextwin = features.context_window(input_arr, left, right)
                print(contextwin.shape)

    def test_compute_deltas(self):
        specgram = np.random.random([1, 400 // 2 + 1, 1000])
        deltas = features.compute_deltas(specgram)
        print(deltas.shape)

    def test_fbank(self):
        inputs = np.random.random([10, 16000])
        feats = features.fbank(inputs)
        print(feats.shape)

    def test_mfcc(self):
        inputs = np.random.random([10, 16000])
        feats = features.mfcc(inputs)
        print(feats.shape)

    def test_complex_norm(self):
        inputs_arr = spectrum.stft(self.test_data, return_complex=False)
        norm = features.complex_norm(inputs_arr)
        print(norm)

    def test_angle(self):
        inputs_arr = spectrum.stft(self.test_data, return_complex=False)
        angle = features.complex_norm(inputs_arr)
        print(angle)


    def test_harmonic(self):
        harm = features.harmonic(self.test_data)
        print(harm)


if __name__ == "__main__":
    test = TestOperators()
    test.setup_method()
    test.test_harmonic()
    test.test_mfcc()


