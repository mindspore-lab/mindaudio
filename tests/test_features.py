import numpy as np
import torch
from speechbrain.processing import features

def test_ContextWindow():
    from mindaudio.data.features.features import ContextWindow
    input_arrs = [np.random.randn(10, 101, 60).astype(dtype=np.float32),
                  np.random.randn(10, 101, 60, 2).astype(dtype=np.float32)]
    left_frames = [3, 4, 5, 0]
    right_frames = [5, 4, 3, 0]
    for left, right in zip(left_frames, right_frames):
        for input_arr in input_arrs:
            in_shape = input_arr.shape
            if len(in_shape) == 3:
                in_tensor = torch.from_numpy(input_arr).transpose(-1, -2)
            else:
                in_tensor = torch.from_numpy(input_arr).permute(0, 3, 2, 1)
            contextwin = ContextWindow(left, right)
            output = contextwin(input_arr)
            contextwin_sp = features.ContextWindow(left, right)
            output_sp = contextwin_sp(in_tensor)
            if len(in_shape) == 3:
                assert np.allclose(output, output_sp.transpose(-1, -2).numpy())
            else:
                assert np.allclose(output, output_sp.permute(0, 3, 2, 1).numpy())
test_ContextWindow()

