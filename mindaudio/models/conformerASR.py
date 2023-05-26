"""Definition of ASR model."""

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from layers.ctc import CTC
from layers.label_smoothing_loss import LabelSmoothingLoss
from transformer.cmvn import GlobalCMVN
from transformer.decoder import BiTransformerDecoder, TransformerDecoder
from transformer.encoder import ConformerEncoder, TransformerEncoder
from utils.cmvn import load_cmvn
from utils.common import IGNORE_ID
from communication.management import get_group_size


class ASRModelWithAcc(nn.Cell):
    """CTC-attention hybrid encoder-decoder model with accuracy computation.

    Args:
        vocab_size (int): Vocabulary size of output layer.
        encoder (ConformerEncoder or TransformerEncoder): Encoder module for ASR model.
        decoder (TransformerDecoder): Decoder module for ASR model.
        ctc (CTC): CTC module for ASR model.
        ctc_weight (float): Weight for CTC loss.
        ignore_id (int): Padded token ID for text.
        reverse_weight (float): Weight for reverse loss.
        lsm_weight (float): Weight for label smoothing loss.
        length_normalized_loss (bool): Whether to do length normalization for loss.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: ConformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.cast = ops.Cast()
        self.div = ops.Div()
        self.equal = ops.Equal()
        self.mul = ops.Mul()
        self.scalar_summary = ops.ScalarSummary()
        self.expand_dims = ops.ExpandDims()
        self.tile = ops.Tile()
        self.topk = ops.TopK()
        self.gather = ops.Gather()
        self.cat = ops.Concat(axis=1)

    def construct(
        self,
        xs_pad: mindspore.Tensor,
        ys_pad: mindspore.Tensor,
        ys_in_pad: mindspore.Tensor,
        ys_out_pad: mindspore.Tensor,
        r_ys_in_pad: mindspore.Tensor,
        r_ys_out_pad: mindspore.Tensor,
        xs_masks: mindspore.Tensor,
        ys_sub_masks: mindspore.Tensor,
        ys_masks: mindspore.Tensor,
        ys_lengths: mindspore.Tensor,
        xs_chunk_masks: mindspore.Tensor,
    ):
        """Encoder + Decoder + Calc loss.

        Args:
            xs_pad (mindspore.Tensor): Padded speech features.
            ys_pad (mindspore.Tensor): Padded text sequences.
            ys_in_pad (mindspore.Tensor): Padded input text for decoder.
            ys_out_pad (mindspore.Tensor): Padded output text for decoder.
            r_ys_in_pad (mindspore.Tensor): Padded input text for right-left decoder.
            r_ys_out_pad (mindspore.Tensor): Padded output text for right-left decoder.
            xs_masks (mindspore.Tensor): Mask of speech features.
            ys_sub_masks (mindspore.Tensor): Mask of text sequences.
            ys_masks (mindspore.Tensor): Mask of text sequences.
            ys_lengths (mindspore.Tensor): Lengths of each text sequences.
            xs_chunk_maks (mindspore.Tensor): Chunk mask of speech features, for streaming ASR.

        Returns:
            tuple: tensor of loss, tensor of attention accuracy
        """
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(xs_pad, xs_masks, xs_chunk_masks)
        encoder_out = self.cast(encoder_out, mstype.float32)
        encoder_mask = self.cast(encoder_mask, mstype.float32)
        encoder_out_lens = self.cast(
            encoder_mask.squeeze().sum(axis=1),
            mstype.int32,
        )

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(
                encoder_out,
                encoder_mask,
                ys_in_pad,
                ys_out_pad,
                r_ys_in_pad,
                r_ys_out_pad,
                ys_masks,
                ys_sub_masks,
            )
        else:
            loss_att = None
            acc_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_lengths)
        else:
            loss_ctc = None

        # 3. final loss
        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        self.scalar_summary("loss", loss)
        if loss_att is not None:
            self.scalar_summary("loss_att", loss_att)
            self.scalar_summary("acc_att", acc_att)
        if loss_ctc is not None:
            self.scalar_summary("loss_ctc", loss_ctc)

        return loss, acc_att

    def _calc_att_loss(
        self,
        encoder_out: mindspore.Tensor,
        encoder_mask: mindspore.Tensor,
        ys_in_pad: mindspore.Tensor,
        ys_out_pad: mindspore.Tensor,
        r_ys_in_pad: mindspore.Tensor,
        r_ys_out_pad: mindspore.Tensor,
        ys_masks: mindspore.Tensor,
        ys_sub_masks: mindspore.Tensor,
    ):
        """Calculate attention loss."""
        # 1. Forward decoder
        decoder_out, r_decoder_out = self.decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_sub_masks, r_ys_in_pad
        )
        decoder_out = self.cast(decoder_out, mstype.float32)
        r_decoder_out = self.cast(r_decoder_out, mstype.float32)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad, ys_masks)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad, ys_masks)
            loss_att = (
                loss_att * (1 - self.reverse_weight) + r_loss_att * self.reverse_weight
            )

        # 3. Compute attention accuracy
        acc_att = self._th_accuracy(
            decoder_out,
            ys_out_pad,
            ys_masks,
        )

        return loss_att, acc_att

    def _th_accuracy(
        self,
        pad_outputs: mindspore.Tensor,
        pad_targets: mindspore.Tensor,
        ys_masks: mindspore.Tensor,
    ):
        """Calculate accuracy.

        Args:
            pad_outputs (mindspore.Tensor): Prediction tensors (B * Lmax, D).
            pad_targets (mindspore.Tensor): Target label tensors (B, Lmax).
            ys_masks (mindspord.Tensor): Target label mask (B, Lmax)

        Returns:
            mindspore.Tensor: Accuracy value (0.0 - 1.0).
        """
        pad_pred = pad_outputs.argmax(2)
        ys_masks = ys_masks.squeeze(1)
        numerator = self.mul(self.equal(pad_pred, pad_targets), ys_masks).sum()
        denominator = ys_masks.sum()
        return self.div(numerator, denominator)


class ASRModel(nn.Cell):
    """CTC-attention hybrid encoder-decoder model.

    Args:
        vocab_size (int): Vocabulary size of output layer.
        encoder (ConformerEncoder or TransformerEncoder): Encoder module for ASR model.
        decoder (TransformerDecoder): Decoder module for ASR model.
        ctc (CTC): CTC module for ASR model.
        ctc_weight (float): Weight for CTC loss.
        ignore_id (int): Padded token ID for text.
        reverse_weight (float): Weight for reverse loss.
        lsm_weight (float): Weight for label smoothing loss.
        length_normalized_loss (bool): Whether to do length normalization for loss.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: ConformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        super().__init__()
        self.acc_net = ASRModelWithAcc(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            ctc_weight=ctc_weight,
            ignore_id=ignore_id,
            reverse_weight=reverse_weight,
            lsm_weight=lsm_weight,
            length_normalized_loss=length_normalized_loss,
        )

    def construct(
        self,
        xs_pad: mindspore.Tensor,
        ys_pad: mindspore.Tensor,
        ys_in_pad: mindspore.Tensor,
        ys_out_pad: mindspore.Tensor,
        r_ys_in_pad: mindspore.Tensor,
        r_ys_out_pad: mindspore.Tensor,
        xs_masks: mindspore.Tensor,
        ys_sub_masks: mindspore.Tensor,
        ys_masks: mindspore.Tensor,
        ys_lengths: mindspore.Tensor,
        xs_chunk_masks: mindspore.Tensor,
    ):
        """Do forward process.

        Args:
            xs_pad (mindspore.Tensor): Padded speech features.
            ys_pad (mindspore.Tensor): Padded text sequences.
            ys_in_pad (mindspore.Tensor): Padded input text for decoder.
            ys_out_pad (mindspore.Tensor): Padded output text for decoder.
            r_ys_in_pad (mindspore.Tensor): Padded input text for right-left decoder.
            r_ys_out_pad (mindspore.Tensor): Padded output text for right-left decoder.
            xs_masks (mindspore.Tensor): Mask of speech features.
            ys_sub_masks (mindspore.Tensor): Mask of text sequences.
            ys_masks (mindspore.Tensor): Mask of text sequences.
            ys_lengths (mindspore.Tensor): Lengths of each text sequences.
            xs_chunk_maks (mindspore.Tensor): Chunk mask of speech features, for streaming ASR.

        Returns:
            tuple: tensor of loss, tensor of attention accuracy
        """
        loss, _ = self.acc_net(
            xs_pad,
            ys_pad,
            ys_in_pad,
            ys_out_pad,
            r_ys_in_pad,
            r_ys_out_pad,
            xs_masks,
            ys_sub_masks,
            ys_masks,
            ys_lengths,
            xs_chunk_masks,
        )
        return loss


def init_asr_model(config, input_dim, vocab_size):
    """Init a ASR model."""
    mean, istd = load_cmvn(config["cmvn_file"], config["is_json_cmvn"])
    global_cmvn = GlobalCMVN(
        mindspore.Tensor(mean, mstype.float32), mindspore.Tensor(istd, mstype.float32)
    )
    if config["mixed_precision"]:
        compute_type = mstype.float16
    else:
        compute_type = mstype.float32

    encoder_type = config["encoder"]
    decoder_type = config["decoder"]
    ctc_weight = config["model_conf"]["ctc_weight"]

    if encoder_type == "conformer":
        encoder = ConformerEncoder(
            input_dim,
            **config["encoder_conf"],
            global_cmvn=global_cmvn,
            compute_type=compute_type,
        )
    elif encoder_type == "transformer":
        encoder = TransformerEncoder(
            input_dim,
            **config["encoder_conf"],
            global_cmvn=global_cmvn,
            compute_type=compute_type,
        )

    else:
        raise NotImplementedError

    if ctc_weight == 1.0:
        decoder = None
    elif decoder_type == "transformer":
        decoder = TransformerDecoder(
            vocab_size,
            encoder.output_size(),
            compute_type=compute_type,
            **config["decoder_conf"],
        )
    elif decoder_type == "bitransformer":
        assert 0.0 < config["model_conf"]["reverse_weight"] < 1.0
        assert config["decoder_conf"]["r_num_blocks"] > 0
        decoder = BiTransformerDecoder(
            vocab_size,
            encoder.output_size(),
            compute_type=compute_type,
            **config["decoder_conf"],
        )
    else:
        raise NotImplementedError

    if ctc_weight == 0.0:
        ctc = None
    else:
        ctc = CTC(vocab_size, encoder.output_size(), compute_type=compute_type)

    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        **config["model_conf"],
    )

    return model


class ASREvalNet(nn.Cell):
    """ASR Eval network."""

    def __init__(self, network):
        super(ASREvalNet, self).__init__()
        self.network = network
        try:
            self.device_num = get_group_size()
            self.all_reduce = ops.AllReduce()
        except (ValueError, RuntimeError):
            self.device_num = 1
            self.all_reduce = None

    def construct(self, *inputs, **kwargs):
        loss = self.network(*inputs, **kwargs)
        if self.all_reduce:
            loss = self.all_reduce(loss)
        return loss / self.device_num
