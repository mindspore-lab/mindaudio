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
"""Prediction net for ASR inference."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np


class PredictNet(nn.Cell):
    """Prediction net for ASR inference.

    Args:
        backbone (nn.Cell): ASR models for inference
        beam_szie (int): size for beam search.
        eos (int): a specific stop token for predict results.
    """

    def __init__(self, backbone, beam_size, eos):
        super(PredictNet, self).__init__()
        self.backbone = backbone
        self.topk = ops.TopK()
        self.tile = ops.Tile()
        self.concat = ops.Concat(1)
        self.zeros_like = ops.ZerosLike()
        self.cast = ops.Cast()
        self.scatter_update = ops.ScatterUpdate()
        self.gather = ops.Gather()
        self.inf = -10000.0
        self.beam_size = beam_size
        self.eos = eos

    def mask_finished_scores(self, score, end_flag):
        """If a sequence is finished, we only allow one alive branch. This
        function aims to give one branch a zero score and the rest -inf score.

        Args:
            score (mindspore.Tensor): A real value array with shape
                (batch_size * beam_size, beam_size).
            flag (mindspore.Tensor): A bool array with shape
                (batch_size * beam_size, 1).

        Returns:
            mindspore.Tensor: (batch_size * beam_size, beam_size).
        """
        beam_size = score.shape[-1]
        zero_mask = self.zeros_like(end_flag)
        if self.beam_size > 1:
            unfinished = self.concat(
                (zero_mask, self.tile(end_flag, (1, beam_size - 1)))
            )
            finished = self.concat((end_flag, self.tile(zero_mask, (1, beam_size - 1))))
        else:
            unfinished = zero_mask
            finished = end_flag
        score = score + unfinished * self.inf
        score = score * (1 - finished)

        return score

    def mask_finished_preds(self, pred, end_flag, eos):
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
        finished = self.cast(self.tile(end_flag, (1, beam_size)), mindspore.int32)
        pred = pred * (1 - finished) + eos * finished
        return pred

    def construct(
        self,
        memory,
        memory_mask,
        tgt,
        tgt_mask,
        valid_length,
        end_flag,
        scores,
        base_index,
    ):
        """Conduct beam search inference.

        Args:
            memory (mindspore.Tensor): encoded speech features.
            memory_mask (mindspore.Tensor): mask of encoded speech features.
            tgt (mindspore.Tensor): init text for inference
            tgt_mask (mindspore.Tensor): mask of init text for inference
            valid_length (mindspore.Tensor): length of decoded text
            end_flag (mindspore.Tensor): flag to indicate whether current speech is finished
            scores (mindspore.Tensor): score for decoded results
            base_index (mindspore.Tensor): base index used in beam search for index shifting

        Returns:
            mindspore.Tensor: decoded results.
        """
        x, _ = self.backbone.acc_net.decoder.embed(tgt)
        for _, decoder in enumerate(self.backbone.acc_net.decoder.decoders):
            x, tgt_mask, memory, memory_mask = decoder(x, tgt_mask, memory, memory_mask)
        if self.backbone.acc_net.decoder.normalize_before:
            y = self.backbone.acc_net.decoder.after_norm(
                self.gather(x, valid_length, 1)
            )
        else:
            y = self.gather(x, valid_length, 1)

        if self.backbone.acc_net.decoder.use_output_layer:
            y = self.backbone.acc_net.decoder.log_softmax(
                self.backbone.acc_net.decoder.output_layer(y)
            )

        # 2.2 First beam prune: select topk best prob at current time
        top_k_logp, top_k_index = self.topk(y, self.beam_size)  # (B*N, N)
        top_k_logp = self.mask_finished_scores(top_k_logp, end_flag)  # (B*N, N)
        top_k_index = self.mask_finished_preds(
            top_k_index, end_flag, self.eos
        )  # (B*N, N)
        batch_size = top_k_logp.shape[0] // self.beam_size
        # 2.3 Second beam prune: select topk score with history
        scores = scores + top_k_logp  # (B*N, N), broadcast add
        scores = scores.view(batch_size, self.beam_size * self.beam_size)  # (B, N*N)
        scores, offset_k_index = self.topk(scores, self.beam_size)  # (B, N)
        scores = scores.view(-1, 1)  # (B*N, 1)

        # 2.4. Compute base index in top_k_index,
        # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
        # then find offset_k_index in top_k_index
        base_k_index = self.tile(base_index, (1, self.beam_size))  # (B, N)
        base_k_index = base_k_index * self.beam_size * self.beam_size
        best_k_index = base_k_index.reshape(-1) + offset_k_index.reshape(-1)  # (B*N)

        # 2.5 Update best hyps
        best_k_pred = top_k_index.reshape(-1)[best_k_index]  # B*N
        # Modify input_ids with newly generated token
        best_hyps_index = best_k_index // self.beam_size
        last_best_k_hyps = tgt[best_hyps_index]
        valid_length += 1
        return last_best_k_hyps, valid_length, end_flag, scores, best_k_pred


class CTCGreedySearch(nn.Cell):
    """CTC greedy search net for ASR inference.

    Args:
        backbone (nn.Cell): ASR models for inference
        pretrained_model (nn.Cell): speech pre-trained models, like wav2vec 2.0.
    """

    def __init__(self, backbone, pretrained_model=False):
        super(CTCGreedySearch, self).__init__()
        self.backbone = backbone
        self.pretrained_model = pretrained_model
        self.encoder = self.backbone.acc_net.encoder
        self.topk = ops.TopK()
        self.cast = ops.Cast()
        self.mul = ops.Mul()

    def construct(self, xs_pad, xs_masks, xs_lengths):
        """Conduct CTC beam search inference.

        Args:
            xs_pad (mindspore.Tensor): input speech samples.
            xs_masks (mindspore.Tensor): mask of input speech samples.
            xs_lengths (mindspore.Tensor): length of input speech samples.

        Returns:
            mindspore.Tensor: decoded results.
        """
        batch_size = xs_pad.shape[0]
        if self.pretrained_model:
            (
                xs_pad,
                xs_masks,
            ) = self.backbone.acc_net.pretrained_model.acc_net.extrator_feature_only(
                xs_pad, xs_masks, xs_lengths
            )
            if self.backbone.acc_net.feature_post_proj:
                xs_pad = self.backbone.acc_net.feature_post_proj(xs_pad)
        xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]
        encoder_out, encoder_mask = self.encoder(xs_pad, xs_masks, xs_masks)
        ctc_probs = self.backbone.acc_net.ctc.compute_log_softmax_out(encoder_out)
        # (B, T, 1)
        topk_prob, topk_index = self.topk(ctc_probs, 1)  # (B, T)
        topk_index = topk_index.view(batch_size, encoder_out.shape[1])
        # masking padded part
        encoder_mask = self.cast(encoder_mask.squeeze(1), mindspore.int32)
        topk_index = self.mul(topk_index, encoder_mask)  # (B, T)
        return topk_index, topk_prob


class Attention(nn.Cell):
    """Attention net for ASR inference.

    Args:
        backbone (nn.Cell): ASR models for inference.
        beam_szie (int): size for beam search.
        eos (int): a specific stop token for predict results.
        pretrained_model (nn.Cell): speech pre-trained models, like wav2vec 2.0.
    """

    def __init__(self, backbone, beam_size, eos, pretrained_model=False):
        super(Attention, self).__init__()
        self.beam_size = beam_size
        self.eos = eos

        self.backbone = backbone
        self.pretrained_model = pretrained_model
        self.predict_model = PredictNet(backbone, beam_size, eos)

        self.tile = ops.Tile()
        self.expand_dims = ops.ExpandDims()
        self.topk = ops.TopK()

        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=1)
        self.scatterupdate = ops.TensorScatterUpdate()
        self.gather = ops.Gather()
        self.beam_index_left = mindspore.Tensor(
            np.arange(beam_size).reshape(-1, 1), mindspore.int32
        )
        self.beam_index_right = mindspore.Tensor(
            np.ones((beam_size, 1)), mindspore.int32
        )
        self.zero = mindspore.Tensor(0, mindspore.int32)
        self.one = mindspore.Tensor(1, mindspore.int32)

    def construct(
        self,
        xs_pad,
        xs_masks,
        xs_lengths,
        hyps_sub_masks,
        input_ids,
        scores,
        end_flag,
        base_index,
    ):
        """Conduct Attention inference.

        Args:
            xs_pad (mindspore.Tensor): input speech samples.
            xs_masks (mindspore.Tensor): mask of input speech samples.
            xs_lengths (mindspore.Tensor): length of input speech samples.
            hyps_sub_masks (mindspore.Tensor):  mask information for hyps.
            input_ids (mindspore.Tensor): infer result.
            scores (mindspore.Tensor): init hyps scores.
            end_flag (mindspore.Tensor): end flags for detect whether the utterance is finished.
            base_index (mindspore.Tensor): end flags for detect whether the utterance is finished.

        Returns:
            mindspore.Tensor: decoded results.
        """
        batch_size = xs_pad.shape[0]
        running_size = batch_size * self.beam_size
        if self.pretrained_model:
            (
                xs_pad,
                xs_masks,
            ) = self.backbone.acc_net.pretrained_model.acc_net.extrator_feature_only(
                xs_pad, xs_masks, xs_lengths
            )
            if self.backbone.acc_net.feature_post_proj:
                xs_pad = self.backbone.acc_net.feature_post_proj(xs_pad)
        xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]
        encoder_out, encoder_mask = self.backbone.acc_net.encoder(
            xs_pad, xs_masks, xs_masks
        )
        maxlen = encoder_out.shape[1]
        encoder_dim = encoder_out.shape[2]
        # (B*N, maxlen, encoder_dim)
        encoder_out = self.tile(
            self.expand_dims(encoder_out, 1), (1, self.beam_size, 1, 1)
        )
        encoder_out = encoder_out.view((running_size, maxlen, encoder_dim))
        # (B*N, 1, max_len)
        encoder_mask = self.tile(
            self.expand_dims(encoder_mask, 1), (1, self.beam_size, 1, 1)
        )
        encoder_mask = encoder_mask.view((running_size, 1, maxlen))
        # 2. Decoder forward step by step
        valid_length = self.zero
        i = self.one
        while i <= maxlen:
            # exit when the length of prediction is larger than maxlen
            if valid_length >= maxlen - 1:
                break
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = self.expand_dims(
                self.gather(self.cast(hyps_sub_masks, mindspore.float32), i, 0), 0
            )
            hyps_mask = self.cast(hyps_mask, mindspore.bool_)
            hyps_mask = self.tile(hyps_mask, (running_size, 1, 1))  # (B*N, i, i)
            input_ids, valid_length, end_flag, scores, best_k_pred = self.predict_model(
                encoder_out,
                encoder_mask,
                input_ids,
                hyps_mask,
                valid_length,
                end_flag,
                scores,
                base_index,
            )
            # 2.6 Update end flag
            indices = self.concat(
                (self.beam_index_left, self.beam_index_right * valid_length)
            )
            input_ids = self.scatterupdate(
                self.cast(input_ids, mindspore.float32),
                indices,
                self.cast(best_k_pred, mindspore.float32),
            )
            input_ids = self.cast(input_ids, mindspore.int32)
            end_flag = self.cast(
                (input_ids[:, valid_length] == self.eos).view(-1, 1), mindspore.float32
            )
            i = i + 1
        # 3. Select best of best
        scores = scores.view(batch_size, self.beam_size)  # (B, N)
        best_scores, best_index = self.topk(scores, 1)  # (B, 1)
        best_hyps_index = best_index + base_index * self.beam_size
        return input_ids, best_hyps_index, best_scores


class CTCPrefixBeamSearch(nn.Cell):
    """CTC prefix beam search net for ASR inference.

    Args:
        backbone (nn.Cell): ASR models for inference.
        pretrained_model (nn.Cell): speech pre-trained models, like wav2vec 2.0.
    """

    def __init__(self, backbone, beam_size, pretrained_model=False):
        super(CTCPrefixBeamSearch, self).__init__()

        self.backbone = backbone
        self.pretrained_model = pretrained_model
        self.beam_size = beam_size
        self.topk = ops.TopK()

    def construct(self, xs_pad, xs_masks, xs_lengths):
        """
        Conduct CTC prefix beam search inference.
        Args:
            xs_pad (mindspore.Tensor): input speech samples.
            xs_masks (mindspore.Tensor): mask of input speech samples.
            xs_lengths (mindspore.Tensor): length of input speech samples.

        Returns:
            mindspore.Tensor: decoded results.
        """
        if self.pretrained_model:
            (
                xs_pad,
                xs_masks,
            ) = self.backbone.acc_net.pretrained_model.acc_net.extrator_feature_only(
                xs_pad, xs_masks, xs_lengths
            )
            if self.backbone.acc_net.feature_post_proj:
                xs_pad = self.backbone.acc_net.feature_post_proj(xs_pad)
        xs_masks = xs_masks[:, :, :-2:2][:, :, :-2:2]
        encoder_out, encoder_mask = self.backbone.acc_net.encoder(
            xs_pad, xs_masks, xs_masks
        )
        ctc_prbs = self.backbone.acc_net.ctc.compute_log_softmax_out(encoder_out)
        ctc_prbs = ctc_prbs.squeeze(0)
        top_k_logp_list, top_k_index_list = self.topk(ctc_prbs, self.beam_size)
        encoder_mask = encoder_mask.squeeze()
        return encoder_out, encoder_mask, top_k_logp_list, top_k_index_list


class AttentionRescoring(nn.Cell):
    """
    Attention rescoring net for ASR inference.
    Conduct rescoring by attention-based decoder.

    Args:
        backbone (nn.Cell): ASR models for inference.
        pretrained_model (nn.Cell): speech pre-trained models, like wav2vec 2.0.
    """

    def __init__(self, backbone, beam_size, pretrained_models=False):
        super(AttentionRescoring, self).__init__()
        self.backbone = backbone
        self.pretrained_models = pretrained_models
        self.beam_size = beam_size
        self.log_softmax = nn.LogSoftmax(axis=-1)

    def construct(self, encoder_out, encoder_mask, hyps_in_pad, hyps_masks):
        """Conduct attention rescoring inference.

        Args:
            encoder_out (mindspore.Tensor): encoded speech features, (beam_size, maxlen_in, feat)
            encoder_mask (mindspore.Tensor): feature mask, (beam_size, 1, maxlen_in)
            hyps_in_pad (mindspore.Tensor): padded input token ids, (beam_size, maxlen_out)
            hyps_masks (mindspore.Tensor): mask for token sequences, (beam_size, maxlen_out, maxlen_out)

        Returns:
            mindspore.Tensor: decoder output.
        """
        decoder_out, _ = self.backbone.acc_net.decoder(
            encoder_out, encoder_mask, hyps_in_pad, hyps_masks
        )
        decoder_out = self.log_softmax(decoder_out)
        return decoder_out
