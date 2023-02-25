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

# This file refers to https://github.com/wenet-e2e/wenet/tree/main/tools/remove_longshortdata.py
"""remove too long or too short data in format.data."""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='remove too long or too short data in format.data')
    parser.add_argument('--data_file',
                        type=str,
                        help='input format data')
    parser.add_argument('--output_data_file',
                        type=str,
                        help='output format data')
    parser.add_argument('--min_input_len', type=float,
                        default=0,
                        help='minimum input seq length, in seconds for raw wav, \
                            in frame numbers for feature data')
    parser.add_argument('--max_input_len', type=float,
                        default=20,
                        help='maximum output seq length, in seconds for raw wav, \
                            in frame numbers for feature data')
    parser.add_argument('--min_output_len', type=float,
                        default=0, help='minimum input seq length, in modeling units')
    parser.add_argument('--max_output_len', type=float,
                        default=500,
                        help='maximum output seq length, in modeling units')
    parser.add_argument('--min_output_input_ratio', type=float, default=0.05,
                        help='minimum output seq length/output seq length ratio')
    parser.add_argument('--max_output_input_ratio', type=float, default=10,
                        help='maximum output seq length/output seq length ratio')
    args = parser.parse_args()

    data_file = args.data_file
    output_data_file = args.output_data_file
    min_input_len = args.min_input_len
    max_input_len = args.max_input_len
    min_output_len = args.min_output_len
    max_output_len = args.max_output_len
    min_output_input_ratio = args.min_output_input_ratio
    max_output_input_ratio = args.max_output_input_ratio

    with open(data_file, 'r') as f, open(output_data_file, 'w') as fout:
        for l in f:
            l = l.strip()
            if l:
                items = l.strip().split('\t')
                token_shape = items[6]
                feature_shape = items[2]
                feat_len = float(feature_shape.split(':')[1].split(',')[0])
                token_len = float(token_shape.split(':')[1].split(',')[0])
                condition = [feat_len > min_input_len,
                             feat_len < max_input_len,
                             token_len > min_output_len,
                             token_len < max_output_len,
                             token_len / feat_len > min_output_input_ratio,
                             token_len / feat_len < max_output_input_ratio,
                             ]
                if all(condition):
                    fout.write('{}\n'.format(l))
                    continue
