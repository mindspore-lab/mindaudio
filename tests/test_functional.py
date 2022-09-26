import numpy as np
import librosa
import torchaudio
import torch
from speechbrain.processing import signal_processing

def test_stft_ifstft():
    from mindaudio.data.features.functional import stft, istft
    data,_ = librosa.load(librosa.ex('trumpet'))
    stft_mat_librosa = librosa.core.stft(data, n_fft=data.shape[0]//128, pad_mode="reflect")
    return_complex = [True, False]
    for return_complex_type in return_complex:
        stft_mat = stft(data, n_fft=data.shape[0]//128, return_complex=return_complex_type)
        if return_complex_type:
            assert np.allclose(stft_mat, stft_mat_librosa)
            reout = istft(stft_mat, n_fft=data.shape[0]//128)
            #TODO:add lenth
            assert np.allclose(data[:reout.shape[-1]], reout, atol= 1e-5)
        else:
            assert np.allclose(stft_mat, np.stack((stft_mat_librosa.real, stft_mat_librosa.imag), -1))

def test_amplitude2db_db2amplitude():
    from mindaudio.data.features.functional import amplitude_to_dB, dB_to_amplitude
    waveforms = np.random.random([10, 120, 40]).astype(dtype=np.float32)
    stypes = ["power","amplitude"]
    top_dbs = [None, 80.0]
    for stype in stypes:
        for top_db in top_dbs:
            amp2db_torchaudio = torchaudio.transforms.AmplitudeToDB(stype=stype,top_db=top_db)
            db_torchaudio = amp2db_torchaudio(torch.tensor(waveforms)).numpy()
            DB = amplitude_to_dB(waveforms,stype=stype,top_db=top_db)
            assert np.allclose(db_torchaudio, DB)
            if stype == "power":
                power = 1
            else:
                power = 0.5
            specgram = dB_to_amplitude(DB, ref=1.0,power=power)
            assert np.allclose(specgram, waveforms, atol=1e-4)


def test_compute_amplitude():
    from mindaudio.data.features.functional import compute_amplitude
    test_inputs = [np.zeros(100),
                   np.random.random(size=(64, 40)),
                   np.random.random(size=(64, 40, 5))]
    for dB in [True, False]:
        if dB:
            scale = "dB"
        else:
            scale = "linear"
        for amp_type in ["avg","peak"]:
            for test_input in test_inputs:
                length = (
                    test_input.shape[1] if len(test_input.shape) > 1 else test_input.shape[0]
                )
                amplitude = compute_amplitude(test_input, length, amp_type, dB)
                amplitude_speechbrain = signal_processing.compute_amplitude(
                    torch.tensor(test_input),length, amp_type, scale)
                assert np.allclose(amplitude, amplitude_speechbrain.numpy())

def test_normalize():
    from mindaudio.data.features.functional import normalize
    input = np.vander(np.arange(-5.0, 5.0))
    output = normalize(input)
    output_librosa = librosa.util.normalize(input)
    assert np.allclose(output, output_librosa)


def test_stereo_to_mono():
    from mindaudio.data.features.functional import stereo_to_mono
    data = np.random.random((15,2))
    mono = stereo_to_mono(data)
    assert np.allclose(mono, np.mean(data, axis=-1))

def test_rescale():
    from mindaudio.data.features.functional import rescale
    data, _ = librosa.load(librosa.ex('trumpet'))
    output = rescale(data, 0.5)
    output_speechbrain = signal_processing.rescale(torch.Tensor(data), torch.Tensor([len(data)]), 0.5)
    assert np.allclose(output, output_speechbrain.numpy())

def test_notch_filter():
    from mindaudio.data.features.functional import notch_filter
    kernel = notch_filter(0.25,)
    kernel_speechbrain = signal_processing.notch_filter(0.25)
    print(kernel)
    assert np.allclose(kernel, kernel_speechbrain)

def test_split():
    from mindaudio.data.features.functional import split
    data, _ = librosa.load(librosa.ex('trumpet'))
    res = split(data)
    res_librosa = librosa.effects.split(data)
    assert np.allclose(res, res_librosa)

def test_trim():
    from mindaudio.data.features.functional import trim
    data, _ = librosa.load(librosa.ex('trumpet'))
    res = trim(data)
    res_librosa = librosa.effects.trim(data)
    assert np.allclose(res[0], res_librosa[0])
    assert np.allclose(res[1], res_librosa[1])

def test_and_reverberate():
    from mindaudio.data.features.functional import reverberate
    data, _ = librosa.load(librosa.ex('trumpet'))
    kernel = np.random.random(size=(4,)).astype(float)
    data2 = np.expand_dims(data, 0)
    kernel2 = np.random.random(size=(1, 4)).astype(float)
    data3 = np.expand_dims(data2, -1)
    kernel3 = np.random.random(size=(1, 4, 1)).astype(float)
    data_list = [data, data2, data3]
    kernel_list = [kernel, kernel2, kernel3]
    for wav, rirwav in zip(data_list, kernel_list):
        res = reverberate(wav, rirwav)
        res_speechbrain = signal_processing.reverberate(torch.Tensor(wav).double(),
                                                        torch.Tensor(rirwav))
        res_speechbrain = res_speechbrain.numpy()
        assert np.allclose(res, res_speechbrain, atol=1e-3)



