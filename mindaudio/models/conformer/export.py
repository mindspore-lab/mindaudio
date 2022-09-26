"""
##############export checkpoint file into air and onnx models#################
python export.py
"""
import argparse
import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from mindaudio.models.conformer.decode.predict_net import Attention, CTCGreedySearch, CTCPrefixBeamSearch
from mindaudio.models.conformer import init_asr_model
from mindaudio.adapter.config import get_config

config = get_config("asr_conformer")
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

def run_export(args_param):
    """run export."""
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
    if args_param.mode == "attention":
        net = Attention(network, config.beam_size, eos)
    elif args_param.mode == "ctc_greedy_search":
        net = CTCGreedySearch(network)
    elif args_param.mode == "ctc_prefix_beam_search":
        net = CTCPrefixBeamSearch(network, config.beam_size)
    elif args_param.mode == "attention_rescoring":
        pass
    else:
        raise ValueError("only support mode ['attention', 'ctc_greedy_search', 'ctc_prefix_beam_search']")

    assert args_param.ckpt is not None, "checkpoint_path is None."

    param_dict = load_checkpoint(args_param.ckpt)
    if args_param.mode == "attention":
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
        export(net, xs_pad, xs_masks, xs_lengths, hyps_sub_masks, input_ids_tensor, scores, end_flag, base_index,
               file_name=args_param.mode, file_format=args_param.export_type)
    elif args_param.mode == "ctc_prefix_beam_search":
        new_dict = {}
        for k, v in param_dict.items():
            new_key = "backbone." + k
            new_dict[new_key] = v
        load_param_into_net(net, new_dict)
        input_x = Tensor(np.zeros([1, 1200, 80], np.float32))
        input_mask = Tensor(np.zeros([1, 1, 1200], np.float32))
        length = Tensor(np.zeros([1], np.float32))
        export(net, input_x, input_mask, length, file_name=args_param.mode, file_format=args_param.export_type)
    elif args_param.mode == "ctc_greedy_search":
        new_dict = {}
        for k, v in param_dict.items():
            new_key = k
            if "encoder" in k:
                new_key = k.replace('acc_net.', '')
            elif "decoder" in k or "ctc" in k:
                new_key = "backbone." + k
            new_dict[new_key] = v
        load_param_into_net(net, new_dict)
        input_x = Tensor(np.zeros([1, 1200, 80], np.float32))
        input_mask = Tensor(np.zeros([1, 1, 1200], np.float32))
        length = Tensor(np.zeros([1], np.float32))
        export(net, input_x, input_mask, length, file_name=args_param.mode, file_format=args_param.export_type)
    else:
        raise ValueError("only support mode ['attention', 'ctc_greedy_search', 'ctc_prefix_beam_search']")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="average model")
    parser.add_argument("--mode", required=True, help="support [attention, ctc_greedy_search, ctc_prefix_beam_search]")
    parser.add_argument("--ckpt", required=True, help="the checkpoint that needs to be converted")
    parser.add_argument("--export_type", type=str, default='MINDIR', help="supported file formats: [MINDIR, ONNX, AIR]")
    args = parser.parse_args()
    run_export(args)
