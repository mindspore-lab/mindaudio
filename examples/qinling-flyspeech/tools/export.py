# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""
##############export checkpoint file into air and onnx models#################
python export.py
"""
import argparse

import mindspore
import numpy as np
from flyspeech.adapter.config import get_config
from flyspeech.decode.predict_net import (
    Attention,
    AttentionRescoring,
    CTCGreedySearch,
    CTCPrefixBeamSearch,
    PredictNet,
)
from flyspeech.model.asr_model import init_asr_model
from mindspore import Tensor, context, export, load_checkpoint, load_param_into_net

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def run_export(args_param):
    """run export."""
    config = get_config(args_param.config_path)
    # load dict
    char_dict = {}
    with open(config.dict, "r") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1
    vocab_size = len(char_dict)
    collate_conf = config.collate_conf
    input_dim = collate_conf.feature_extraction_conf.mel_bins
    network = init_asr_model(config, input_dim, vocab_size)
    assert args_param.decode_ckpt is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args_param.decode_ckpt)
    # Ascend310 infer only supported not full_graph now for attention decode.
    is_full_graph = False
    if args_param.decode_mode == "attention":
        if is_full_graph:
            net = Attention(network, config.beam_size, eos)
        else:
            net = PredictNet(network, config.beam_size, eos)
        _export_attention(param_dict, net, args_param, is_full_graph)
    elif args_param.decode_mode == "ctc_greedy_search":
        net = CTCGreedySearch(network)
        _export_ctc_greedy_search(param_dict, net, args_param)
    elif args_param.decode_mode == "ctc_prefix_beam_search":
        net = CTCPrefixBeamSearch(network, config.beam_size)
        _export_ctc_prefix_beam_search(param_dict, net, args_param)
    elif args_param.decode_mode == "attention_rescoring":
        net_ctc = CTCPrefixBeamSearch(network, config.beam_size)
        net_rescore = AttentionRescoring(network, config.beam_size)
        _export_attention_rescoring(param_dict, net_ctc, net_rescore, args_param)
    else:
        raise ValueError(
            "only support mode ['attention', 'ctc_greedy_search', 'ctc_prefix_beam_search',"
            "attention_rescoring]"
        )


def _export_ctc_greedy_search(param_dict, net, args_param):
    """export ctc_greedy_search"""
    new_dict = {}
    for k, v in param_dict.items():
        new_key = k
        if "encoder" in k:
            new_key = k.replace("acc_net.", "")
        elif "decoder" in k or "ctc" in k:
            new_key = "backbone." + k
        if "network" in new_key:
            new_key = new_key.replace("network.", "")
        new_dict[new_key] = v
    load_param_into_net(net, new_dict)
    input_x = Tensor(np.zeros([1, 1200, 80], np.float32))
    input_mask = Tensor(np.zeros([1, 1, 1200], np.float32))
    length = Tensor(np.zeros([1], np.float32))
    export(
        net,
        input_x,
        input_mask,
        length,
        file_name=args_param.decode_mode,
        file_format=args_param.export_type,
    )


def _export_ctc_prefix_beam_search(param_dict, net, args_param):
    """export ctc_prefix_beam_search"""
    new_dict = {}
    for k, v in param_dict.items():
        new_key = "backbone." + k
        if "network" in new_key:
            new_key = new_key.replace("network.", "")
        new_dict[new_key] = v
    load_param_into_net(net, new_dict)
    input_x = Tensor(np.zeros([1, 1200, 80], np.float32))
    input_mask = Tensor(np.zeros([1, 1, 1200], np.float32))
    length = Tensor(np.zeros([1], np.float32))
    export(
        net,
        input_x,
        input_mask,
        length,
        file_name=args_param.decode_mode,
        file_format=args_param.export_type,
    )


def _export_attention(param_dict, net, args_param, is_full_graph):
    """export attention"""
    # for full_graph, Ascend310 infer not supported now.
    if is_full_graph:
        new_dict = {}
        for k, v in param_dict.items():
            new_key = "predict_model.backbone." + k
            new_dict[new_key] = v
        load_param_into_net(net, new_dict)
        xs_pad = Tensor(np.zeros([1, 1200, 80], np.float32))
        xs_masks = Tensor(np.zeros([1], np.int32))
        xs_lengths = Tensor(np.zeros([300, 299, 299], np.bool))
        hyps_sub_masks = Tensor(np.zeros([10, 299], np.int32))
        input_ids_tensor = Tensor(np.zeros([10, 299], np.int32))
        scores = Tensor(np.zeros([10, 1], np.float32))
        end_flag = Tensor(np.zeros([10, 1], np.float32))
        base_index = Tensor(np.zeros([1, 1], np.float32))
        export(
            net,
            xs_pad,
            xs_masks,
            xs_lengths,
            hyps_sub_masks,
            input_ids_tensor,
            scores,
            end_flag,
            base_index,
            file_name=args_param.decode_mode,
            file_format=args_param.export_type,
        )
    # for not full_graph, Ascend310 infer supported now.
    else:
        # for encoder
        new_dict_encoder = {}
        for k, v in param_dict.items():
            if "encoder" in k:
                new_key = "backbone." + k
                if "network" in new_key:
                    new_key = new_key.replace("network.", "")
                new_dict_encoder[new_key] = v

        encoder_net = net.backbone.acc_net.encoder
        load_param_into_net(encoder_net, new_dict_encoder)
        xs_pad = Tensor(np.zeros([1, 1200, 80], np.float32))
        xs_masks = Tensor(np.zeros([1, 1, 299], np.float32))
        export(
            encoder_net,
            xs_pad,
            xs_masks,
            xs_masks,
            file_name=args_param.decode_mode + "_encoder",
            file_format=args_param.export_type,
        )

        # for predict
        new_dict_predict = {}
        for k, v in param_dict.items():
            new_key = "backbone." + k
            if "network" in new_key:
                new_key = new_key.replace("network.", "")
            new_dict_predict[new_key] = v
        load_param_into_net(net, new_dict_predict)
        encoder_out_np = Tensor(np.zeros([10, 299, 256], np.float32))
        encoder_mask_np = Tensor(np.zeros([10, 1, 299], np.float32))
        inputs_ids = Tensor(np.zeros([10, 299], np.int32))
        hyps_mask = Tensor(np.zeros([10, 299, 299], np.float32))
        valid_length = Tensor(0, mindspore.int32)
        end_flag = Tensor(np.zeros([10, 1], np.float32))
        scores = Tensor(np.zeros([10, 1], np.float32))
        base_index = Tensor(np.zeros([1, 1], np.int32))
        export(
            net,
            encoder_out_np,
            encoder_mask_np,
            inputs_ids,
            hyps_mask,
            valid_length,
            end_flag,
            scores,
            base_index,
            file_name=args_param.decode_mode + "_predict",
            file_format=args_param.export_type,
        )


def _export_attention_rescoring(param_dict, net_ctc, net_rescore, args_param):
    """export attention_rescoring"""
    # export for ctc
    new_dict_ctc = {}
    for k, v in param_dict.items():
        new_key = "backbone." + k
        if "network" in new_key:
            new_key = new_key.replace("network.", "")
        new_dict_ctc[new_key] = v
    load_param_into_net(net_ctc, new_dict_ctc)
    input_x = Tensor(np.zeros([1, 1200, 80], np.float32))
    input_mask = Tensor(np.zeros([1, 1, 1200], np.float32))
    length = Tensor(np.zeros([1], np.float32))
    export(
        net_ctc,
        input_x,
        input_mask,
        length,
        file_name=args_param.decode_mode + "_ctc",
        file_format=args_param.export_type,
    )

    # export for rescore
    new_dict_rescore = {}
    for k, v in param_dict.items():
        new_key = "backbone." + k
        if "network" in new_key:
            new_key = new_key.replace("network.", "")
        new_dict_rescore[new_key] = v
    load_param_into_net(net_rescore, new_dict_rescore)
    encoder_out = Tensor(np.zeros([10, 299, 256], np.float32))
    encoder_mask = Tensor(np.zeros([10, 1, 299], np.float32))
    hyps_in_pad = Tensor(np.zeros([10, 31], np.int32))
    hyps_sub_masks = Tensor(np.zeros([10, 31, 31], np.float32))
    export(
        net_rescore,
        encoder_out,
        encoder_mask,
        hyps_in_pad,
        hyps_sub_masks,
        file_name=args_param.decode_mode + "_rescore",
        file_format=args_param.export_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export model")
    parser.add_argument(
        "--decode_mode",
        required=True,
        help="support [attention, ctc_greedy_search,"
        "ctc_prefix_beam_search,attention_rescoring]",
    )
    parser.add_argument(
        "--decode_ckpt", required=True, help="the checkpoint that needs to be converted"
    )
    parser.add_argument(
        "--export_type",
        type=str,
        default="MINDIR",
        help="supported file formats: [MINDIR, ONNX, AIR]",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/asr_conformer.yaml",
        help="config file",
    )
    parser.add_argument(
        "--cmvn_file", type=str, default="/path/train/global_cmvn", help="cmvn file"
    )
    parser.add_argument(
        "--dict", type=str, default="/path/dict/lang_char.txt", help="dict file"
    )
    args = parser.parse_args()
    run_export(args)
