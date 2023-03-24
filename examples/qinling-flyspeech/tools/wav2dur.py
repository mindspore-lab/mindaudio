#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# This file refers to https://github.com/wenet-e2e/wenet/tree/main/tools/wav2dur.py

"""Calculate times."""
import sys

from mindaudio.data.io import read

scp = sys.argv[1]
dur_scp = sys.argv[2]

with open(scp, "r") as f, open(dur_scp, "w") as fout:
    cnt = 0
    total_duration = 0
    for l in f:
        items = l.strip().split()
        wav_id = items[0]
        fname = items[1]
        cnt += 1
        waveform, rate = read(fname)
        frames = len(waveform)
        duration = frames / float(rate)
        total_duration += duration
        fout.write("{} {}\n".format(wav_id, duration))
    print("process {} utts".format(cnt))
    print("total {} s".format(total_duration))
