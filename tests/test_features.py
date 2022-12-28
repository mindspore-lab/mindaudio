import numpy as np
import mindaudio.data.io as io
import mindaudio.data.features as features
import mindaudio.data.spectrum as spectrum

def test_spectral_centroid():
    waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
    spectralcentroid = features.spectral_centroid(waveform, sr)  # (channel, time)
    print(spectralcentroid.shape)

def test_context_window():
    input_arrs = [np.random.randn(10, 101, 60).astype(dtype=np.float32),
                  np.random.randn(10, 101, 60, 2).astype(dtype=np.float32)]
    left_frames = [3, 4, 5, 0]
    right_frames = [5, 4, 3, 0]
    for left, right in zip(left_frames, right_frames):
        for input_arr in input_arrs:
            contextwin = features.context_window(input_arr, left, right)
            print(contextwin.shape)

def test_compute_deltas():
    specgram = np.random.random([1, 400 // 2 + 1, 1000])
    deltas = features.compute_deltas(specgram)
    print(deltas.shape)


def test_fbank():
    inputs = np.random.random([10, 16000])
    feats = features.fbank(inputs)
    print(feats.shape)

def test_mfcc():
    inputs = np.random.random([10, 16000])
    feats = features.mfcc(inputs)
    print(feats.shape)

def test_complex_norm():
    waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
    inputs_arr = spectrum.stft(waveform, return_complex=False)
    norm = features.complex_norm(inputs_arr)
    print(norm)

def test_angle():
    waveform, sr = io.read('./samples/ASR/BAC009S0002W0122.wav')
    inputs_arr = spectrum.stft(waveform, return_complex=False)
    angle = features.complex_norm(inputs_arr)
    print(angle)


if __name__ == "__main__":
    test_spectral_centroid()
    test_context_window()
    test_compute_deltas()
    test_fbank()
    test_mfcc()
    test_angle()


