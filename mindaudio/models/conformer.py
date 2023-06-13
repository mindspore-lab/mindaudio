"""Definition of ASR model."""

from typing import Optional, Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

from .layers.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from .layers.convolution import ConvolutionModule
from .layers.dense import Dense
from .layers.embedding import (
    ConvPositionalEncoding,
    NoPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
)
from .layers.layernorm import LayerNorm
from .layers.positionwise_feed_forward import PositionwiseFeedForward
from .layers.subsampling import Conv2dSubsampling4
from .layers.swish import Swish


class ConformerEncoderLayer(nn.Cell):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Cell,
        feed_forward: nn.Cell,
        feed_forward_macaron: nn.Cell,
        conv_module: nn.Cell,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
        compute_type=mstype.float32,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.ff_scale = 0.5
        self.norm_ff = LayerNorm(size, epsilon=1e-5)
        self.norm_mha = LayerNorm(size, epsilon=1e-5)
        self.norm_ff_macaron = LayerNorm(size, epsilon=1e-5)
        self.norm_conv = LayerNorm(size, epsilon=1e-5)
        self.norm_final = LayerNorm(size, epsilon=1e-5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Dense(size + size, size).to_float(compute_type)
        self.cat_1 = ops.Concat(1)
        self.cat_f1 = ops.Concat(-1)
        self.cast = ops.Cast()
        self.get_dtype = ops.DType()
        self.compute_type = compute_type

    def construct(
        self,
        x: mindspore.Tensor,
        mask: mindspore.Tensor,
        pos_emb: mindspore.Tensor,
        mask_pad: mindspore.Tensor,
        output_cache: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Compute encoded features.

        Args:
            x (minspore.Tensor): (#batch, time, size)
            mask (minspore.Tensor): Mask tensor for the input (#batch, 1, time).
            pos_emb (minspore.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (mindspore.Tensor): mask for input tensor.
            output_cache (minspore.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
        Returns:
            minspore.Tensor: Output tensor (#batch, time, size).
            minspore.Tensor: Mask tensor (#batch, time).
        """
        # Macaron-Net Feedforward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff_macaron(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
        if not self.normalize_before:
            x = self.norm_ff_macaron(x)

        # Multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if output_cache is None:
            x_q = x
        else:
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, mask, pos_emb)

        if self.concat_after:
            x_concat = self.cat_f1((x, self.cast(x_att, x.dtype)))
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution module
        residual = x
        if self.normalize_before:
            x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x, mask_pad))
        if not self.normalize_before:
            x = self.norm_conv(x)

        # Feedforward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        # Final normalization
        x = self.norm_final(x)

        if output_cache is not None:
            x = self.cat_1([output_cache, x], dim=1)

        return x, mask


class BaseEncoder(nn.Cell):
    """Base encode instance.

    Args:
        input_size (int): input dim
        output_size (int): dimension of attention
        positional_dropout_rate (float): dropout rate after adding
            positional encoding
        input_layer (str): input layer type.
            optional [linear, conv2d, conv2d6, conv2d8]
        pos_enc_layer_type (str): Encoder positional encoding layer type.
            opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        feature_norm (bool): whether do feature norm to input features, like CMVN
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        positional_dropout_rate: float = 0.1,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        feature_norm: bool = True,
        global_cmvn: mindspore.nn.Cell = None,
        compute_type=mindspore.float32,
    ):
        """construct BaseEncoder."""
        super().__init__()
        self._output_size = output_size

        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "conv_pos":
            pos_enc_class = ConvPositionalEncoding
        else:
            pos_enc_class = NoPositionalEncoding
        self.input_layer = input_layer
        if input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
            self.embed = subsampling_class(
                input_size,
                output_size,
                pos_enc_class(output_size, positional_dropout_rate),
                compute_type,
            )
        else:
            self.embed = pos_enc_class(output_size, positional_dropout_rate)

        self.normalize_before = normalize_before
        if normalize_before:
            self.after_norm = LayerNorm(output_size, epsilon=1e-5)

        self.feature_norm = feature_norm
        self.global_cmvn = global_cmvn

    def output_size(self) -> int:
        return self._output_size

    def construct(
        self,
        xs: mindspore.Tensor,
        masks: mindspore.Tensor,
        xs_chunk_masks: mindspore.Tensor = None,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            masks: masks for the input xs ()
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: mindspore.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        if self.global_cmvn:
            xs = self.global_cmvn(xs)

        # masks is subsampled to (B, 1, T/subsample_rate)
        xs, pos_emb = self.embed(xs)
        for layer in self.encoders:
            xs, xs_chunk_masks = layer(xs, xs_chunk_masks, pos_emb, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


class ConformerEncoder(BaseEncoder):
    """conformer encoder module.

    Args:
        input_size (int): input dim
        output_size (int): dimension of attention
        attention_heads (int): the number of heads of multi head attention
        linear_units (int): the hidden units number of position-wise feed
            forward
        num_blocks (int): the number of decoder blocks
        dropout_rate (float): dropout rate
        attention_dropout_rate (float): dropout rate in attention
        positional_dropout_rate (float): dropout rate after adding
            positional encoding
        input_layer (str): input layer type.
            optional [linear, conv2d, conv2d6, conv2d8]
        pos_enc_layer_type (str): Encoder positional encoding layer type.
            opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
        normalize_before (bool):
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        feature_norm (bool): whether do feature norm to input features, like CMVN
        concat_after (bool): whether to concat attention layer's input
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        activation_type (str): type of activation type.
        cnn_module_kernel (int): kernel size for CNN module
        cnn_module_norm (str): normalize type for CNN module, batch norm or layer norm.
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        feature_norm: bool = True,
        concat_after: bool = False,
        activation_type: str = "relu",
        cnn_module_kernel: int = 15,
        cnn_module_norm: str = "batch_norm",
        global_cmvn: mindspore.nn.Cell = None,
        compute_type=mstype.float32,
    ):
        """Construct ConformerEncoder."""
        super().__init__(
            input_size,
            output_size,
            positional_dropout_rate,
            input_layer,
            pos_enc_layer_type,
            normalize_before,
            feature_norm,
            global_cmvn,
            compute_type,
        )

        activation = Swish()

        # self-attention module definition
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention

        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            compute_type,
        )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
            compute_type,
        )

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            output_size,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            1,
            True,
            compute_type,
        )

        self.encoders = nn.CellList(
            [
                ConformerEncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args),  # Macron-Net style
                    convolution_layer(*convolution_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    compute_type,
                )
                for _ in range(num_blocks)
            ]
        )


class DecoderLayer(nn.Cell):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (nn.Cell): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (nn.Cell): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (nn.Cell): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """

    def __init__(
        self,
        size: int,
        self_attn: nn.Cell,
        src_attn: nn.Cell,
        feed_forward: nn.Cell,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
        compute_type=mstype.float32,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size, epsilon=1e-12)
        self.norm2 = LayerNorm(size, epsilon=1e-12)
        self.norm3 = LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = Dense(size + size, size).to_float(compute_type)
            self.concat_linear2 = Dense(size + size, size).to_float(compute_type)
        self.cat1 = ops.Concat(axis=-1)
        self.cat2 = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.cast = ops.Cast()

    def construct(
        self,
        tgt: mindspore.Tensor,
        tgt_mask: mindspore.Tensor,
        memory: mindspore.Tensor,
        memory_mask: mindspore.Tensor,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Compute decoded features.

        Args:
            tgt (mindspore.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (mindspore.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (mindspore.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (mindspore.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (mindspore.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).

        Returns:
            mindspore.Tensor: Output tensor (#batch, maxlen_out, size).
            mindspore.Tensor: Mask for output tensor (#batch, maxlen_out).
            mindspore.Tensor: Encoded memory (#batch, maxlen_in, size).
            mindspore.Tensor: Encoded memory mask (#batch, maxlen_in).
        """
        # Self-attention module
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        tgt_q = tgt
        tgt_q_mask = tgt_mask
        if self.concat_after:
            tgt_concat = self.cat1(
                (
                    tgt_q,
                    self.cast(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask), tgt_q.dtype),
                )
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # Src-attention module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        if self.concat_after:
            x_concat = self.cat1(
                (x, self.cast(self.src_attn(x, memory, memory, memory_mask), x.dtype))
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        # Feedforward module
        residual = x
        if self.normalize_before:
            x = self.norm3(x)

        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        return x, tgt_mask, memory, memory_mask


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
        activation = nn.ReLU()
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
