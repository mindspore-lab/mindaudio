import numpy as np
import mindaudio.data.io as io
import mindaudio.data.spectrum as spectrum
def test_amplitude_to_dB():
    waveforms = np.random.random([1, 400 // 2 + 1, 30])
    out = spectrum.amplitude_to_dB(waveforms)
    print(out.shape)

def test_dB_to_amplitude():
    specgram = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
    out = spectrum.dB_to_amplitude(specgram, 0.5, 0.5)
    print(out.shape)

def test_stft():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    matrix = spectrum.stft(waveform)
    print(matrix.shape)

def test_istft():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    print(waveform.shape)
    matrix = spectrum.stft(waveform)
    res = spectrum.istft(matrix)
    print(res.shape)

def test_compute_amplitude():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    amp_avg = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='avg')
    print(amp_avg)
    amp_peak = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='peak')
    print(amp_peak)
    amp_db = spectrum.compute_amplitude(waveform, lengths=waveform.shape[0], amp_type='peak', dB=True)
    print(amp_db)

def test_spectrogram():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    spec = spectrum.spectrogram(waveform)
    print(spec.shape)

def test_melspectrogram():
    waveform, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    spec = spectrum.melspectrogram(waveform)
    print(spec.shape)

def test_magphase():
    waveforms, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    D = spectrum.stft(waveforms)
    magnitude, phase = spectrum.magphase(D, power=2.0, iscomplex=True)
    print(magnitude, phase)

def test_melscale():
    waveforms, _ = io.read('./samples/ASR/BAC009S0002W0122.wav')
    spec = spectrum.spectrogram(waveforms, n_fft=1024)
    melscale_spec = spectrum.melscale(spec, n_stft=1024 // 2 + 1)

if __name__ == "__main__":
    test_amplitude_to_dB()
    test_dB_to_amplitude()
    test_stft()
    test_istft()
    test_compute_amplitude()
    test_spectrogram()
    test_melspectrogram()
    test_magphase()
    test_melscale()