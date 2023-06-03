"""Decoder definition."""

from typing import Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from layers.dense import Dense
from layers.layernorm import LayerNorm
from modules.attention import MultiHeadedAttention
from modules.decoder_layer import DecoderLayer
from modules.embedding import PositionalEncoding
from modules.positionwise_feed_forward import PositionwiseFeedForward
from utils.net import get_activation


class TransformerDecoder(nn.Cell):
    """Base class of Transformer decoder module.

    Args:
        vocab_size (int): output dim.
        encoder_output_size (int): dimension of attention.
        attention_heads (int): the number of heads of multi head attention.
        linear_units (int): the hidden units number of position-wise feedforward.
        num_blocks (int): the number of decoder blocks.
        dropout_rate (float): dropout rate.
        positional_dropout_rate (float): dropout rate positional encoding.
        self_attention_dropout_rate (float): dropout rate for self-attention.
        src_attention_dropout_rate (float): dropout rate for src-attention.
        input_layer (str): input layer type.
        use_output_layer (bool): whether to use output layer.
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after (bool): whether to concat attention layer's input and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
        compute_type=mstype.float32,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        activation = get_activation("relu")
        self.first_flag = True

        if input_layer == "embed":
            self.embed = nn.SequentialCell(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.use_output_layer = use_output_layer

        if normalize_before:
            self.after_norm = LayerNorm(attention_dim, epsilon=1e-12)

        if use_output_layer:
            self.output_layer = Dense(attention_dim, vocab_size).to_float(compute_type)

        self.decoders = nn.CellList(
            [
                DecoderLayer(
                    attention_dim,
                    MultiHeadedAttention(
                        attention_heads,
                        attention_dim,
                        self_attention_dropout_rate,
                        compute_type,
                    ),
                    MultiHeadedAttention(
                        attention_heads,
                        attention_dim,
                        src_attention_dropout_rate,
                        compute_type,
                    ),
                    PositionwiseFeedForward(
                        attention_dim,
                        linear_units,
                        dropout_rate,
                        activation,
                        compute_type,
                    ),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    compute_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.expand_dims = ops.ExpandDims()
        self.log_softmax = nn.LogSoftmax()
        self.tensor0 = mindspore.Tensor((0,))

    # pylint: disable=W0613
    def construct(
        self,
        memory: mindspore.Tensor,
        memory_mask: mindspore.Tensor,
        ys_in_pad: mindspore.Tensor,
        ys_masks: mindspore.Tensor,
        r_ys_in_pad: mindspore.Tensor = None,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            ys_masks: mask for token sequences
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                vocab_size) if use_output_layer is True,
                self.tensor0, in order to unify api with bidirectional decoder
        """
        x, _ = self.embed(ys_in_pad)
        for layer in self.decoders:
            x, ys_masks, memory, memory_mask = layer(x, ys_masks, memory, memory_mask)

        if self.normalize_before:
            x = self.after_norm(x)

        if self.use_output_layer:
            x = self.output_layer(x)

        return x, self.tensor0


class BiTransformerDecoder(nn.Cell):
    """Base class of Transformer decoder module.

    Args:
        vocab_size (int): output dim.
        encoder_output_size (int): dimension of attention.
        attention_heads (int): the number of heads of multi head attention.
        linear_units (int): the hidden units number of position-wise feedforward.
        num_blocks (int): the number of decoder blocks.
        r_num_blocks: the number of right to left decoder blocks.
        dropout_rate (float): dropout rate.
        positional_dropout_rate (float): dropout rate positional encoding.
        self_attention_dropout_rate (float): dropout rate for self-attention.
        src_attention_dropout_rate (float): dropout rate for src-attention.
        input_layer (str): input layer type.
        use_output_layer (bool): whether to use output layer.
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after (bool): whether to concat attention layer's input and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
        compute_type=mstype.float32,
    ):
        super().__init__()
        self.left_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            use_output_layer,
            normalize_before,
            concat_after,
            compute_type,
        )
        self.right_decoder = TransformerDecoder(
            vocab_size,
            encoder_output_size,
            attention_heads,
            linear_units,
            r_num_blocks,
            dropout_rate,
            positional_dropout_rate,
            self_attention_dropout_rate,
            src_attention_dropout_rate,
            input_layer,
            use_output_layer,
            normalize_before,
            concat_after,
            compute_type,
        )

    def construct(
        self,
        memory: mindspore.Tensor,
        memory_mask: mindspore.Tensor,
        ys_in_pad: mindspore.Tensor,
        ys_masks: mindspore.Tensor,
        r_ys_in_pad: mindspore.Tensor,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Forward decoder.

        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out), use for right to left decoder
            ys_masks: mask for token sequences
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                vocab_size) if use_output_layer is True,
                r_x: decoded token score before softmax (batch, maxlen_out,
                vocab_size) if use_output_layer is True, use for right to left decoder

        """
        l_x, _ = self.left_decoder(memory, memory_mask, ys_in_pad, ys_masks)
        r_x, _ = self.right_decoder(memory, memory_mask, r_ys_in_pad, ys_masks)
        return l_x, r_x
