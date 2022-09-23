import os
import sys

import numpy as np
import mindspore.context as context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

sys.path.append('.')
from mindaudio.adapter.config import get_config
from mindaudio.adapter.log import get_logger
from mindaudio.adapter.parallel_info import get_device_id
from mindaudio.examples.conformer.dataset import create_asr_predict_dataset, load_language_dict
from mindaudio.models.conformer.src.decode.predict_net import Attention, CTCGreedySearch, CTCPrefixBeamSearch, PredictNet
from mindaudio.models.conformer.src.decode.recognize import ctc_greedy_search, ctc_prefix_beam_search, recognize
from mindaudio.models.conformer.src.model.asr_model import init_asr_model

logger = get_logger()
config = get_config('asr_config')


def main():
    """main function for asr_predict."""
    exp_dir = config.exp_name
    decode_mode = config.decode_mode
    model_dir = os.path.join(exp_dir, 'model')
    decode_ckpt = os.path.join(model_dir, config.decode_ckpt)
    decode_dir = os.path.join(exp_dir, 'test_' + decode_mode)
    os.makedirs(decode_dir, exist_ok=True)
    result_file = open(os.path.join(decode_dir, 'result.txt'), 'w')

    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=get_device_id())

    # load test data
    test_dataset = create_asr_predict_dataset(config.test_data, config.dataset_conf, config.collate_conf)
    # load dict
    sos, eos, vocab_size, char_dict = load_language_dict(config.dict)

    collate_conf = config.collate_conf
    input_dim = collate_conf.feature_extraction_conf.mel_bins

    # define network
    network = init_asr_model(config, input_dim, vocab_size)
    param_dict = load_checkpoint(decode_ckpt)
    load_param_into_net(network, param_dict)
    logger.info('Successfully loading the asr model: %s', decode_ckpt)
    network.set_train(False)

    if config.decode_mode == 'ctc_greedy_search':
        model = Model(CTCGreedySearch(network))
    elif config.decode_mode == 'attention' and config.full_graph:
        model = Model(Attention(network, config.beam_size, eos))
    elif config.decode_mode == 'attention' and not config.full_graph:
        model = Model(PredictNet(network, config.beam_size, eos))
    elif config.decode_mode == 'ctc_prefix_beam_search':
        model = Model(CTCPrefixBeamSearch(network, config.beam_size))
    tot_sample = test_dataset.get_dataset_size()
    logger.info('Total predict samples size: %d', tot_sample)
    count = 0
    for data in test_dataset:
        uttid, xs_pad, xs_masks, tokens, xs_lengths = data
        logger.info('Using decoding strategy: %s', config.decode_mode)
        if config.decode_mode == 'attention':
            start_token = np.array([sos], np.int32)
            scores = np.array([0.0] + [-float('inf')] * (config.beam_size - 1), np.float32)
            end_flag = np.array([0.0] * config.beam_size, np.float32)
            base_index = np.array(np.arange(xs_pad.shape[0]), np.int32).reshape(-1, 1)

            hyps, _ = recognize(
                model,
                xs_pad,  # (1, 500, 80)
                xs_masks,  # (1, 1, 500)
                start_token,  # [4232]
                base_index,  # [[0]]
                scores,  # [ 0.00000000e+00, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf]
                end_flag,  # [0]*10
                config.beam_size,
                eos,
                xs_lengths,
                config.full_graph,
            )
            hyps = [hyp.tolist() for hyp in hyps]
        elif config.decode_mode == 'ctc_greedy_search':
            hyps, _ = ctc_greedy_search(model, xs_pad, xs_masks, xs_lengths)
        # ctc_prefix_beam_search and attention_rescoring restrict the batch_size = 1
        # and return one result in List[int]. Here change it to List[List[int]] for
        # compatible with other batch decoding mode
        elif config.decode_mode == 'ctc_prefix_beam_search':
            assert xs_pad.shape[0] == 1
            hyp, _ = ctc_prefix_beam_search(model, xs_pad, xs_masks, config.beam_size, xs_lengths)
            hyps = [hyp]
        else:
            raise NotImplementedError

        # batch size equals to 1
        content = ''
        ground_truth = ''
        count += 1
        for w in hyps[0]:
            if w == eos:
                break
            content += char_dict[w]
        tokens = tokens.asnumpy()
        for w in tokens:
            ground_truth += char_dict[w]
        logger.info('Labs (%d/%d): %s %s', count, tot_sample, uttid, ground_truth)
        logger.info('Hyps (%d/%d): %s %s', count, tot_sample, uttid, content)
        result_file.write('{} {}\n'.format(uttid, content))
        result_file.flush()

    result_file.close()


if __name__ == '__main__':
    main()
