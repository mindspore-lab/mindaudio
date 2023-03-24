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
# This file was refer to project https://github.com/wenet-e2e/wenet.git
# ============================================================================
"""Apply beam search on attention decoder."""

from collections import defaultdict

import numpy as np
from flyspeech.utils.common import (
    add_sos_eos,
    log_add,
    pad_sequence,
    remove_duplicates_and_blank,
)
from infer.infer_ascend_python.utils import make_pad_mask, subsequent_mask, topk_fun


def recognize(
    model_encoder,
    model_predict,
    xs_pad,
    xs_masks,
    start_token,
    base_index,
    scores,
    end_flag,
    beam_size,
    eos,
):
    """Apply beam search on attention decoder
    Args:
        xs_pad (numpy.ndarray): (batch, max_len, feat_dim)
        xs_masks (numpy.ndarray): (batch, )
        start_token (numpy.ndarray): start token <sos> for decoder input
        base_index (numpy.ndarray): base index for select the topK predicted tokens
        scores (numpy.ndarray): init hyps scores
        end_flag (numpy.ndarray): end flags for detect whether the utterance is finished
        beam_size (int): beam size for beam search
        eos (int): the <eos> token
    Returns:
        numpy.ndarray: decoding result, (batch, max_result_len)
        numpy.ndarray: hyps scores, (batch)
    """
    batch_size = xs_pad.shape[0]
    xs_pad = xs_pad.astype(np.float32)
    xs_masks = xs_masks.astype(np.float32)
    # Assume B = batch_size and N = beam_size
    # 1. Encoder
    xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]
    inputs = model_encoder.get_inputs()
    outputs = model_encoder.get_outputs()
    inputs[0].set_data_from_numpy(xs_pad)
    inputs[1].set_data_from_numpy(xs_masks)
    inputs[2].set_data_from_numpy(xs_masks)
    model_encoder.predict(inputs, outputs)
    encoder_out = outputs[0].get_data_to_numpy()
    encoder_mask = outputs[1].get_data_to_numpy()
    maxlen = encoder_out.shape[1]
    encoder_dim = encoder_out.shape[2]
    running_size = batch_size * beam_size

    # (B*N, maxlen, encoder_dim)
    encoder_out_np = np.tile(np.expand_dims(encoder_out, 1), (1, beam_size, 1, 1))
    encoder_out_np = encoder_out_np.reshape((running_size, maxlen, encoder_dim))
    # (B*N, 1, max_len)
    encoder_mask_np = np.tile(np.expand_dims(encoder_mask, 1), (1, beam_size, 1, 1))
    encoder_mask_np = encoder_mask_np.reshape((running_size, 1, maxlen))

    hyps_sub_masks = [[[False]]]
    for i in range(1, maxlen + 1):
        hyps_sub_masks.append(
            np.pad(
                subsequent_mask(i),
                ((0, maxlen - i), (0, maxlen - i)),
                "constant",
                constant_values=(False, False),
            )
        )

    # (B*N, 1)
    hyps = np.tile(np.expand_dims(start_token, 1), (running_size, 1))
    pad_length = maxlen - start_token.shape[-1]
    input_ids = np.pad(
        hyps, ((0, 0), (0, pad_length)), "constant", constant_values=(0, 0)
    )
    # (B*N, 1)
    scores = np.expand_dims(np.tile(scores, (batch_size,)), 1)
    # (B*N, 1)
    end_flag = np.expand_dims(np.tile(end_flag, (batch_size,)), 1)

    # 2. Decoder forward step by step
    valid_length = np.array(0).astype(np.int32)
    for i in range(1, maxlen + 1):
        # exit when the length of prediction is larger than maxlen
        if valid_length >= maxlen - 1:
            break
        # Stop if all batch and all beam produce eos
        if end_flag.sum() == running_size:
            break
        # 2.1 Forward decoder step
        hyps_mask = np.expand_dims(hyps_sub_masks[i], 0)
        hyps_mask = np.tile(hyps_mask, (running_size, 1, 1)).astype(
            np.float32
        )  # (B*N, i, i)
        inputs = model_predict.get_inputs()
        outputs = model_predict.get_outputs()
        inputs[0].set_data_from_numpy(encoder_out_np)
        inputs[1].set_data_from_numpy(encoder_mask_np)
        inputs[2].set_data_from_numpy(input_ids)
        inputs[3].set_data_from_numpy(hyps_mask)
        inputs[4].set_data_from_numpy(valid_length)
        inputs[5].set_data_from_numpy(end_flag)
        inputs[6].set_data_from_numpy(scores)
        inputs[7].set_data_from_numpy(base_index)
        model_predict.predict(inputs, outputs)
        input_ids = outputs[0].get_data_to_numpy()
        valid_length = outputs[1].get_data_to_numpy()
        scores = outputs[3].get_data_to_numpy()
        best_k_pred = outputs[4]
        input_ids[:, valid_length] = best_k_pred.get_data_to_numpy()
        # 2.6 Update end flag
        end_flag = (input_ids[:, valid_length] == eos).astype(np.float32).reshape(-1, 1)
    # 3. Select best of best
    scores = scores.reshape(batch_size, beam_size)  # (B, N)
    # TODO: length normalization
    best_scores, best_index = topk_fun(scores, 1)  # (B, 1)
    best_hyps_index = best_index + base_index * beam_size

    best_hyps = input_ids[best_hyps_index.squeeze(0)]
    best_hyps = best_hyps[:, 1:]
    return best_hyps, best_scores.squeeze(0)


def ctc_greedy_search(model, xs_pad, xs_masks):
    """Apply greedy search on encoder
    Args:
        xs_pad (numpy.ndarray): (batch, max_len, feat_dim)
        xs_mask (numpy.ndarray): (batch, )
    Returns:
        numpy.ndarray: decoding result, (batch, max_result_len)
    """
    xs_pad = xs_pad.astype(np.float32)  # (B, T, D)
    xs_masks = xs_masks.astype(np.float32)  # (B, 1, T)
    inputs = model.get_inputs()
    outputs = model.get_outputs()
    inputs[0].set_data_from_numpy(xs_pad)
    inputs[1].set_data_from_numpy(xs_masks)
    model.predict(inputs, outputs)
    topk_index = outputs[0]
    hyps = [hyp.tolist() for hyp in topk_index.get_data_to_numpy()]
    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
    return hyps


def ctc_prefix_beam_search(model, xs_pad, xs_masks, beam_size):
    """Apply CTC prefix beam search
    Args:
        xs_pad (numpy.ndarray): (batch, max_len, feat_dim)
        xs_mask (numpy.ndarray): (batch, )
    Returns:
        Tuple(Tuple[int], float): decoding result and its score
    """
    xs_pad = xs_pad.astype(np.float32)  # (B, T, D)
    xs_masks = xs_masks.astype(np.float32)  # (B, 1, T)
    batch_size = xs_pad.shape[0]
    # for CTC prefix beam search, only support batch_size = 1
    assert batch_size == 1
    inputs = model.get_inputs()
    outputs = model.get_outputs()
    inputs[0].set_data_from_numpy(xs_pad)
    inputs[1].set_data_from_numpy(xs_masks)
    model.predict(inputs, outputs)

    encoder_out = outputs[0]
    encoder_mask = outputs[1]
    top_k_logp_list = outputs[2]
    top_k_index_list = outputs[3]
    maxlen = encoder_out.get_data_to_numpy().shape[1]
    encoder_mask = encoder_mask.get_data_to_numpy().tolist()
    top_k_logp_list = top_k_logp_list.get_data_to_numpy().tolist()
    top_k_index_list = top_k_index_list.get_data_to_numpy().tolist()
    # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
    cur_hyps = [(tuple(), (0.0, -float("inf")))]
    # 3. CTC prefix beam search step by step
    for t in range(0, maxlen):
        if encoder_mask[t] == 0:
            continue
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float("inf"), -float("inf")))
        # 2.1 First beam prune: select topk best
        top_k_logp = top_k_logp_list[t]
        top_k_index = top_k_index_list[t]
        for index, s in enumerate(top_k_index):
            ps = top_k_logp[index]
            for prefix, (pb, pnb) in cur_hyps:
                prefix_len = len(prefix)
                last = prefix[-1] if prefix_len > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
        # 3.2 Second beam prune
        next_hyps = sorted(
            next_hyps.items(), key=lambda x: log_add(list(x[1])), reverse=True
        )
        cur_hyps = next_hyps[:beam_size]
    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]

    return hyps, encoder_out, encoder_mask


def attention_rescoring(
    model_ctc,
    model_rescore,
    xs_pad,
    xs_masks,
    sos,
    eos,
    beam_size,
    ctc_weight,
    max_tgt_len=30,
):
    """Apply attention rescoring decoding
    Args:
        xs_pad (numpy.ndarray): (batch, max_len, feat_dim)
        xs_mask (numpy.ndarray): (batch, )
        beam_size (int): beam size for beam search
    Returns:
        Tuple(Tuple[int], float): decoding result and its score
    """
    # 1.1 ctc prefix beamsearch
    # len(hyps) = beam_size, encoder_out.shape = (1, maxlen, encoder_dim)
    hyps, encoder_out, encoder_mask = ctc_prefix_beam_search(
        model_ctc, xs_pad, xs_masks, beam_size
    )
    assert len(hyps) == beam_size
    hyps_lens = np.array([len(hyp[0]) for hyp in hyps])

    # generate the input sequence for ASR decoder
    # hyps_in: add <sos> label to the hyps
    hyps_in, _ = add_sos_eos([np.array(hyp[0]) for hyp in hyps], sos, eos)
    # the padding_max_len should be increase by 1, since y is paded with a <sos>
    hyps_in_pad = pad_sequence(
        hyps_in,
        batch_first=True,
        padding_value=eos,
        padding_max_len=max_tgt_len + 1,
        atype=np.int32,
    )

    hyps_in_pad = hyps_in_pad.astype(np.int32)
    hyps_mask = np.expand_dims(
        ~make_pad_mask(hyps_lens + 1, max_len=max_tgt_len + 1), 1
    )
    m = np.expand_dims(subsequent_mask(max_tgt_len + 1), 0)
    hyps_sub_masks = (hyps_mask & m).astype(np.float32)
    encoder_out = encoder_out.get_data_to_numpy()
    encoder_out = encoder_out.repeat(beam_size, axis=0)
    encoder_mask = (
        np.array([[encoder_mask]]).repeat(beam_size, axis=0).astype(np.float32)
    )
    inputs = model_rescore.get_inputs()
    outputs = model_rescore.get_outputs()
    inputs[0].set_data_from_numpy(encoder_out)
    inputs[1].set_data_from_numpy(encoder_mask)
    inputs[2].set_data_from_numpy(hyps_in_pad)
    inputs[3].set_data_from_numpy(hyps_sub_masks)
    model_rescore.predict(inputs, outputs)
    decoder_out = outputs[0]
    decoder_out = decoder_out.get_data_to_numpy()
    # only use decoder score for rescoring
    best_score = -float("inf")
    best_index = 0
    for i, hyp in enumerate(hyps):
        if len(hyp[0]) > max_tgt_len + 1:
            continue
        score = 0.0
        for j, w in enumerate(hyp[0]):
            score += decoder_out[i][j][w]
        score += decoder_out[i][len(hyp[0])][eos]
        # add ctc score
        score += hyp[1] * ctc_weight
        if score > best_score:
            best_score = score
            best_index = i
    return hyps[best_index][0], best_score
