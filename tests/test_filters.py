import os
import sys

sys.path.append('.')
import mindaudio.data.filters as filters
import mindaudio.data.io as io
from mindaudio.data.augment import convolve1d


class TestOperators():
    def setup_method(self):
        self.root_path = sys.path[0]
        self.data_path = os.path.join(
            self.root_path, 'samples', 'ASR', 'BAC009S0002W0122.wav'
        )

    def test_notch_filter(self):
        waveform, sr = io.read(self.data_path)
        kernel = filters.notch_filter(0.25)
        notched_signals = convolve1d(waveform, kernel)
        print(notched_signals.shape)

    def test_low_pass_filter(self):
        waveform, sample_rate = io.read(self.data_path)
        cutoff_freq = 1500
        out_waveform = filters.low_pass_filter(
            waveform, sample_rate, cutoff_freq
        )
        print(out_waveform)

    def test_peaking_equalizer(self):
        waveform, sample_rate = io.read(self.data_path)
        center_freq = 1500
        gain = 3.0
        quality_factor = 0.707
        out_waveform = filters.peaking_equalizer(
            waveform, sample_rate, center_freq, gain, quality_factor
        )
        print(out_waveform)


if __name__ == "__main__":
    test = TestOperators()
    test.setup_method()
    test.test_notch_filter()
    test.test_low_pass_filter()
    test.test_peaking_equalizer()
