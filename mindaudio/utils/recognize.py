"""Apply beam search on attention decoder."""

from collections import defaultdict

import mindspore.common.dtype as mstype
import numpy as np
from mindspore import Tensor

from .common import add_sos_eos, log_add, pad_sequence, remove_duplicates_and_blank
from .mask import make_pad_mask, subsequent_mask


def mask_finished_scores(score, end_flag):
    """If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (mindspore.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (mindspore.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        mindspore.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.shape[-1]
    zero_mask = np.zeros_like(end_flag)
    if beam_size > 1:
        unfinished = np.concatenate(
            (zero_mask, np.tile(end_flag, (1, beam_size - 1))), axis=1
        )
        finished = np.concatenate(
            (end_flag, np.tile(zero_mask, (1, beam_size - 1))), axis=1
        )
    else:
        unfinished = zero_mask
        finished = end_flag
    score = np.add(score, np.multiply(unfinished, -10000.0))
    score = np.multiply(score, (1 - finished))

    return score


def mask_finished_preds(pred, end_flag, eos):
    """If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (mindspore.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (mindspore.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        mindspore.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.shape[-1]
    finished = np.tile(end_flag, (1, beam_size)).astype(np.int32)
    pred = pred * (1 - finished) + eos * finished
    return pred


def topk_fun(logits, topk=5):
    """Get topk."""
    batch_size, _ = logits.shape
    value = []
    index = []
    for i in range(batch_size):
        target_column = logits[i].tolist()
        sorted_array = [(k, v) for k, v in enumerate(target_column)]
        sorted_array.sort(key=lambda x: x[1], reverse=True)
        topk_array = sorted_array[:topk]
        index_tmp, value_tmp = zip(*topk_array)
        value.append(value_tmp)
        index.append(index_tmp)
    return np.array(value), np.array(index)


def recognize(
    model,
    xs_pad,  # (1, 500, 80)
    xs_masks,  # (1, 1, 500)
    start_token,
    base_index,
    scores,
    end_flag,
    beam_size,
    eos,
    xs_lengths,
    full_graph=False,
    pretrained_model=False,
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
        pretrained_mode (bool): whether to use the features extracted from self-supervised
            pretrained model
    Returns:
        numpy.ndarray: decoding result, (batch, max_result_len)
        numpy.ndarray: hyps scores, (batch)
    """
    batch_size = xs_pad.shape[0]
    xs_pad = Tensor(xs_pad, mstype.float32)  # (1, 500, 80)
    xs_masks = Tensor(xs_masks, mstype.float32)  # (1, 1, 500)
    if full_graph:
        maxlen = xs_pad.shape[1] // 4 - 1
        running_size = batch_size * beam_size
        hyps_sub_masks = [np.zeros((maxlen, maxlen)).astype(np.bool_)]
        for i in range(1, maxlen + 1):
            hyps_sub_masks.append(
                np.pad(
                    subsequent_mask(i),
                    ((0, maxlen - i), (0, maxlen - i)),
                    "constant",
                    constant_values=(False, False),
                )
            )

        hyps = np.tile(np.expand_dims(start_token, 1), (running_size, 1))  # (10, 1)
        pad_length = maxlen - start_token.shape[-1]  # 124-1=123
        input_ids = np.pad(
            hyps, ((0, 0), (0, pad_length)), "constant", constant_values=(0, 0)
        )

        hyps_sub_masks = Tensor(hyps_sub_masks)
        input_ids_tensor = Tensor(input_ids)
        scores = Tensor(np.expand_dims(np.tile(scores, (batch_size,)), 1))
        end_flag = Tensor(np.expand_dims(np.tile(end_flag, (batch_size,)), 1))
        base_index = Tensor(base_index)

        input_ids, best_hyps_index, best_scores = model.predict(
            xs_pad,
            xs_masks,
            xs_lengths,
            hyps_sub_masks,
            input_ids_tensor,
            scores,
            end_flag,
            base_index,
        )
        best_hyps_index = best_hyps_index.asnumpy()
        best_scores = best_scores.asnumpy()
        input_ids = input_ids.asnumpy()
    else:
        # Assume B = batch_size and N = beam_size
        # 1. Encoder
        if pretrained_model:
            (
                xs_pad,
                xs_masks,
            ) = model.predict_network.backbone.acc_net.pretrained_model.acc_net.extrator_feature_only(
                xs_pad, xs_masks, xs_lengths
            )
            if model.predict_network.backbone.acc_net.feature_post_proj:
                xs_pad = model.predict_network.backbone.acc_net.feature_post_proj(
                    xs_pad
                )
        xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]
        encoder_out, encoder_mask = model.predict_network.backbone.acc_net.encoder(
            xs_pad, xs_masks, xs_masks
        )  # (1, 124, 256)  (1, 1, 124)
        maxlen = encoder_out.shape[1]  # 124
        encoder_dim = encoder_out.shape[2]  # 256
        running_size = batch_size * beam_size  # 1 * 10

        # (B*N, maxlen, encoder_dim)
        encoder_out_np = np.tile(
            np.expand_dims(encoder_out.asnumpy(), 1), (1, beam_size, 1, 1)
        )  # (1, 10, 124, 256)
        encoder_out_np = encoder_out_np.reshape(
            (running_size, maxlen, encoder_dim)
        )  # (10, 124, 256)
        # (B*N, 1, max_len)
        encoder_mask_np = np.tile(
            np.expand_dims(encoder_mask.asnumpy(), 1), (1, beam_size, 1, 1)
        )
        encoder_mask_np = encoder_mask_np.reshape(
            (running_size, 1, maxlen)
        )  # (10, 1, 124)

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
        hyps = np.tile(np.expand_dims(start_token, 1), (running_size, 1))  # (10, 1)
        pad_length = maxlen - start_token.shape[-1]  # 124-1=123
        input_ids = np.pad(
            hyps, ((0, 0), (0, pad_length)), "constant", constant_values=(0, 0)
        )
        # (B*N, 1)
        scores = Tensor(np.expand_dims(np.tile(scores, (batch_size,)), 1))  # (10, 1)
        # (B*N, 1)
        end_flag = np.expand_dims(np.tile(end_flag, (batch_size,)), 1)  # (10, 1)

        # 2. Decoder forward step by step
        valid_length = 0
        for i in range(1, maxlen + 1):
            # exit when the length of prediction is larger than maxlen
            if valid_length >= maxlen - 1:
                break
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = np.expand_dims(hyps_sub_masks[i], 0)
            hyps_mask = np.tile(hyps_mask, (running_size, 1, 1))  # (B*N, i, i)
            if i == 1:
                model.predict_network.add_flags_recursive(is_first_iteration=True)
            else:
                model.predict_network.add_flags_recursive(is_first_iteration=False)

            input_ids, valid_length, end_flag, scores, best_k_pred = model.predict(
                Tensor(encoder_out_np),
                Tensor(encoder_mask_np),
                Tensor(input_ids),
                Tensor(hyps_mask),
                Tensor(valid_length),
                Tensor(end_flag),
                scores,
                Tensor(base_index),
            )
            input_ids = input_ids.asnumpy()
            valid_length = valid_length.asnumpy()
            input_ids[:, valid_length] = best_k_pred.asnumpy()
            # 2.6 Update end flag
            end_flag = (
                (input_ids[:, valid_length] == eos).astype(np.float32).reshape(-1, 1)
            )
        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)  # (B, N)
        # TODO: length normalization
        best_scores, best_index = topk_fun(scores.asnumpy(), 1)  # (B, 1)
        best_hyps_index = best_index + base_index * beam_size

    best_hyps = input_ids[best_hyps_index.squeeze(0)]
    best_hyps = best_hyps[:, 1:]
    return best_hyps, best_scores.squeeze(0)


def ctc_greedy_search(model, xs_pad, xs_masks, xs_lengths):
    """Apply greedy search on encoder
    Args:
        xs_pad (numpy.ndarray): (batch, max_len, feat_dim)
        xs_mask (numpy.ndarray): (batch, )
    Returns:
        numpy.ndarray: decoding result, (batch, max_result_len)
    """
    # 1. Encoder
    xs_pad = Tensor(xs_pad, mstype.float32)  # (B, T, D)
    xs_masks = Tensor(xs_masks, mstype.float32)  # (B, 1, T)
    topk_index, topk_prob = model.predict(xs_pad, xs_masks, xs_lengths)
    hyps = [hyp.tolist() for hyp in topk_index.asnumpy()]
    scores = topk_prob.max(1)
    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]

    return hyps, scores


def ctc_prefix_beam_search(model, xs_pad, xs_masks, beam_size, xs_lengths):
    """Apply CTC prefix beam search
    Args:
        xs_pad (numpy.ndarray): (batch, max_len, feat_dim)
        xs_mask (numpy.ndarray): (batch, )
        beam_size (int): beam size for beam search
    Returns:
        Tuple(Tuple[int], float): decoding result and its score
    """
    xs_pad = Tensor(xs_pad, mstype.float32)  # (B, T, D)
    xs_masks = Tensor(xs_masks, mstype.float32)  # (B, 1, T)
    batch_size = xs_pad.shape[0]
    # for CTC prefix beam search, only support batch_size = 1
    assert batch_size == 1
    encoder_out, encoder_mask, top_k_logp_list, top_k_index_list = model.predict(
        xs_pad, xs_masks, xs_lengths
    )
    maxlen = encoder_out.asnumpy().shape[1]
    encoder_mask = encoder_mask.asnumpy().tolist()
    top_k_logp_list = top_k_logp_list.asnumpy().tolist()
    top_k_index_list = top_k_index_list.asnumpy().tolist()
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
    xs_lengths,
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
        model_ctc, xs_pad, xs_masks, beam_size, xs_lengths
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

    hyps_in_pad = Tensor(hyps_in_pad, mstype.int32)
    hyps_mask = np.expand_dims(
        ~make_pad_mask(hyps_lens + 1, max_len=max_tgt_len + 1), 1
    )
    m = np.expand_dims(subsequent_mask(max_tgt_len + 1), 0)
    hyps_sub_masks = (hyps_mask & m).astype(np.float32)
    hyps_sub_masks = Tensor(hyps_sub_masks)

    encoder_out = encoder_out.repeat(beam_size, axis=0)
    encoder_mask = Tensor([[encoder_mask]]).repeat(beam_size, axis=0)
    decoder_out = model_rescore.predict(
        encoder_out, encoder_mask, hyps_in_pad, hyps_sub_masks
    )

    # only use decoder score for rescoring
    best_score = -float("inf")
    best_index = 0
    for i, hyp in enumerate(hyps):
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
