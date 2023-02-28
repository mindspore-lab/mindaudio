# Copyright 2022 Huawei Technologies Co., Ltd
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

""" ASR inference process.

python infer.py --config_path <CONFIG_FILE>
"""

import os
import time
import argparse
import logging
import glob
import numpy as np

import mindspore_lite as mslite

from mindaudio.data.io import read
from flyspeech.adapter.parallel_info import get_device_id
from flyspeech.adapter.config import get_config
from flyspeech.dataset.feature import compute_fbank_feats
from flyspeech.utils.common import pad_sequence
from infer.infer_ascend_python.utils import load_language_dict, get_padding_length, make_pad_mask
from infer.infer_ascend_python.recognize import ctc_greedy_search, ctc_prefix_beam_search, recognize, attention_rescoring

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def data_preprocess_asr(wav_path, collate_conf, frame_bucket_limit):
    """process wav files and change them to tensors"""
    waveform, sample_rate = read(wav_path)
    waveform = waveform * (1 << 15)

    xs = compute_fbank_feats(
        waveform,
        sample_rate,
        mel_bin=int(collate_conf.feature_extraction_conf["mel_bins"]),
        frame_len=int(
            collate_conf.feature_extraction_conf["frame_length"]),
        frame_shift=int(
            collate_conf.feature_extraction_conf["frame_shift"]),
    )

    # batch size equals to 1
    # pad wavs
    padding_length = get_padding_length(xs.shape[0], frame_bucket_limit)
    xs_pad = pad_sequence(
        [xs],
        batch_first=True,
        padding_value=0.0,
        padding_max_len=padding_length,
        atype=np.float32,
    )
    xs_lengths = np.array([x.shape[0] for x in [xs]], dtype=np.int32)
    xs_masks = np.expand_dims(
        ~make_pad_mask(xs_lengths, max_len=padding_length), 1)
    return xs_pad, xs_masks, xs_lengths


def run_infer(args_param):
    """main function for asr_predict"""
    config = get_config(args_param.config_path)
    exp_dir = config.exp_name
    decode_mode = config.decode_mode
    decode_dir = os.path.join(exp_dir, decode_mode)
    os.makedirs(decode_dir, exist_ok='True')
    ascend_device_info = mslite.AscendDeviceInfo(device_id=get_device_id())
    cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
    context = mslite.Context(thread_num=1, thread_affinity_mode=2)
    context.append_device_info(ascend_device_info)
    context.append_device_info(cpu_device_info)

    # load network
    if config.decode_mode == 'ctc_greedy_search' or config.decode_mode == 'ctc_prefix_beam_search':
        model_path = args_param.infer_model_path_1
        model = mslite.Model()
        model.build_from_file(model_path, mslite.ModelType.MINDIR_LITE, context)
    # Ascend310 infer only supported not full_graph now for attention decode.
    elif config.decode_mode == 'attention':
        encoder_model_path = args_param.infer_model_path_1
        predict_model_path = args_param.infer_model_path_2
        encoder_model = mslite.Model()
        predict_model = mslite.Model()
        encoder_model.build_from_file(encoder_model_path, mslite.ModelType.MINDIR_LITE, context)
        predict_model.build_from_file(predict_model_path, mslite.ModelType.MINDIR_LITE, context)
    elif config.decode_mode == 'attention_rescoring':
        ctc_model_path = args_param.infer_model_path_1
        rescore_model_path = args_param.infer_model_path_2
        ctc_model = mslite.Model()
        rescore_model = mslite.Model()
        ctc_model.build_from_file(ctc_model_path, mslite.ModelType.MINDIR_LITE, context)
        rescore_model.build_from_file(rescore_model_path, mslite.ModelType.MINDIR_LITE, context)

    # load dict
    sos, eos, _, char_dict = load_language_dict(config.dict)

    # load test data
    frame_bucket_limit = [int(i) for i in config["test_dataset_conf"]["frame_bucket_limit"].split(",")]
    file_dir_path = args_param.infer_data_path
    files = glob.glob(os.path.join(file_dir_path, "**/*.wav"))
    print("total wav files: ", len(files))
    file_idx = 0
    result_file = open(os.path.join(decode_dir, 'result.txt'), 'w')
    count = 0
    while True:
        start_time = time.time()
        wav_path = files[file_idx]
        uttid = os.path.basename(wav_path)[:-4]
        print(uttid)
        data = data_preprocess_asr(wav_path, config.collate_conf, frame_bucket_limit)
        xs_pad, xs_masks, _ = data
        logging.info("Using decoding strategy: %s", config.decode_mode)

        if config.decode_mode == "ctc_greedy_search":
            hyps = ctc_greedy_search(model, xs_pad, xs_masks)

        elif config.decode_mode == "ctc_prefix_beam_search":
            assert xs_pad.shape[0] == 1
            hyps, _, _ = ctc_prefix_beam_search(
                model, xs_pad, xs_masks, config.beam_size)
            hyps = [hyps[0][0]]

        elif config.decode_mode == "attention":
            start_token = np.array([sos], np.int32)
            scores = np.array([0.0] + [-float('inf')] * (config.beam_size - 1), np.float32)
            end_flag = np.array([0.0] * config.beam_size, np.float32)
            base_index = np.array(np.arange(xs_pad.shape[0]), np.int32).reshape(-1, 1)
            hyps, _ = recognize(encoder_model, predict_model, xs_pad, xs_masks, start_token,
                                base_index, scores, end_flag, config.beam_size, eos)
        elif config.decode_mode == "attention_rescoring":
            assert xs_pad.shape[0] == 1
            hyps, _ = attention_rescoring(ctc_model, rescore_model, xs_pad, xs_masks,
                                          sos, eos, config.beam_size, config.ctc_weight,
                                          config.test_dataset_conf.token_max_length)
            hyps = [hyps]
        else:
            raise NotImplementedError

        # batch size equals to 1
        content = ''
        count += 1
        for w in hyps[0]:
            if w == eos:
                break
            content += char_dict[w]
        logging.info('Hyps (%d): %s %s', count, uttid, content)
        end_time = time.time()
        print("{} cost {}".format(uttid, end_time - start_time))

        result_file.write('{} {}\n'.format(uttid, content))
        result_file.flush()

        file_idx += 1
        if file_idx >= len(files):
            break
    result_file.close()

    print("============ end ================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model infer")
    parser.add_argument("--infer_model_path_1", required=True, default='./',
                        help="support [ctc_greedy_search, ctc_prefix_beam_search, attention, attention_rescoring]")
    parser.add_argument("--infer_model_path_2", default='./',
                        help="support [attention, attention_rescoring]")
    parser.add_argument("--infer_data_path", required=True, help="The folder path of the .wav file")
    parser.add_argument("--config_path", type=str, default='config/asr_conformer.yaml', help="config path")
    parser.add_argument("--decode_mode", type=str, default='ctc_greedy_search', help="decode mode")
    args = parser.parse_args()
    run_infer(args)
