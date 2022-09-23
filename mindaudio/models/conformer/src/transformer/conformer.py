from typing import Optional
from typing import Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

from flyspeech.transformer.attention import MultiHeadedAttention
from flyspeech.transformer.attention import RelPositionMultiHeadedAttention
from flyspeech.transformer.convolution import ConvolutionModule
from flyspeech.transformer.positionwise_feed_forward import PositionwiseFeedForward
from flyspeech.transformer.embedding import PositionalEncoding
from flyspeech.layers.swish import Swish
from flyspeech.layers.layernorm import LayerNorm
from flyspeech.layers.dense import Dense


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
            dropout_rate: float = 0.1,
            concat_after: bool = False,
            attention_heads: int = 4,
            output_size: int = 256,
            attention_dropout_rate: float = 0.0,
            attention_type: str = "rel_pos",
            cnn_module_kernel: int = 15,
            activation: nn.Cell = Swish(),
            cnn_module_norm: str = "batch_norm",
            linear_units: int = 2048,
            compute_type=mstype.float32,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.ff_scale = 0.5
        self.dropout = nn.Dropout(1 - dropout_rate)
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = Dense(size + size, size).to_float(compute_type)
        self.cat_1 = ops.Concat(1)
        self.cat_f1 = ops.Concat(-1)
        self.cast = ops.Cast()
        self.norm_mha = LayerNorm(size, epsilon=1e-5)
        self.norm_conv = LayerNorm(size, epsilon=1e-5)
        self.norm_final = LayerNorm(size, epsilon=1e-5)

        # self-attention module definition
        if attention_type == "abs_pos":
            self.mha_layer = MultiHeadedAttention(
                attention_heads,
                output_size,
                attention_dropout_rate,
                compute_type,
            )
        elif attention_type == "rel_pos":
            self.mha_layer = RelPositionMultiHeadedAttention(
                attention_heads,
                output_size,
                attention_dropout_rate,
                compute_type,
            )
        else:
            raise NotImplementedError

        self.conv_module = ConvolutionModule(
            output_size, cnn_module_kernel, activation, cnn_module_norm, 1, True, compute_type
        )

        self.ffn_module1 = nn.SequentialCell([
            LayerNorm(size),
            PositionwiseFeedForward
            (
                output_size,
                linear_units,
                dropout_rate,
                activation,
                compute_type,
            ),
            nn.Dropout(1-dropout_rate),
        ])

        self.ffn_module2 = nn.SequentialCell([
            LayerNorm(size),
            PositionwiseFeedForward
            (
                output_size,
                linear_units,
                dropout_rate,
                activation,
                compute_type,
            ),
            nn.Dropout(1 - dropout_rate),
        ])

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

        x = x + self.ff_scale * self.ffn_module1(x)

        # Multi-headed self-attention module
        residual = x
        x = self.norm_mha(x)
        if output_cache is None:
            x_q = x
        else:
            # TODO: wait to be reviewed
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.mha_layer(x_q, x, x, mask, pos_emb)

        # TODO: need to be reviewed
        if self.concat_after:
            x_concat = self.cat_f1((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        # Convolution module
        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv_module(x, mask_pad))

        # Feedforward module
        x = self.norm_final(x + self.ff_scale * self.ffn_module2(x))

        if output_cache is not None:
            x = self.cat_1([output_cache, x], dim=1)

        return x, mask

class ConformerEncoder(nn.Cell):
    """Transformer encoder module.

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
            activation_type: str = "Swish",
            cnn_module_kernel: int = 15,
            cnn_module_norm: str = "batch_norm",
            compute_type=mstype.float32,
    ):
        """Construct ConformerEncoder"""
        super().__init__(input_size, output_size, positional_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         feature_norm, compute_type)


        self.encoders = nn.CellList([
            ConformerEncoderLayer(
                size=input_size,
                dropout_rate=dropout_rate,
                concat_after=False,
                compute_type=compute_type,
                attention_heads=attention_heads,
                output_size=output_size,
                attention_dropout_rate=attention_dropout_rate,
                attention_type=pos_enc_layer_type,
                cnn_module_kernel=cnn_module_kernel,
                activation=Swish,
                cnn_module_norm=cnn_module_norm,
                linear_units=linear_units,
            ) for _ in range(num_blocks)
        ])
    def output_size(self) -> int:
        return self._output_size

    def construct(
            self,
            xs: mindspore.Tensor,
            masks: mindspore.Tensor,
            xs_chunk_masks: mindspore.Tensor = None
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        xs, pos_emb = self.embed(xs)
        for layer in self.encoders:
            xs, xs_chunk_masks = layer(
                xs,
                xs_chunk_masks,
                pos_emb,
                masks
            )
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks


class TransformerDecoderLayer(nn.Cell):
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
        concat_after (bool): Whether to concat attention layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
        compute_type (dtype): whether to use mix precision training.
    """
    def __init__(
            self,
            size: int,
            attention_dim,
            self_attention_dropout_rate,
            attention_heads,
            linear_units,
            activation,
            dropout_rate: float,
            normalize_before: bool = True,
            compute_type=mstype.float32,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()

        self.self_attn = MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    compute_type,
                )

        self.src_attn = MultiHeadedAttention(
                    attention_heads,
                    attention_dim,
                    self_attention_dropout_rate,
                    compute_type,
                )

        self.feed_forward = PositionwiseFeedForward(
                    attention_dim,
                    linear_units,
                    dropout_rate,
                    activation,
                    compute_type,
                )

        # normalization layers
        self.size = size
        self.norm1 = LayerNorm(size, epsilon=1e-12)
        self.norm2 = LayerNorm(size, epsilon=1e-12)
        self.norm3 = LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout_rate)

        self.normalize_before = normalize_before
        self.cat1 = ops.Concat(axis=-1)
        self.cat2 = ops.Concat(axis=1)
        #self.expand_dims = ops.ExpandDims()

    def construct(
            self,
            tgt: mindspore.Tensor,
            tgt_mask: mindspore.Tensor,
            memory: mindspore.Tensor,
            memory_mask: mindspore.Tensor,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor,
               mindspore.Tensor]:
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

        # tgt_q = tgt
        # tgt_q_mask = tgt_mask

        x = residual + self.dropout(
            self.self_attn(tgt, tgt, tgt, tgt_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # Src-attention module
        residual = x
        if self.normalize_before:
            x = self.norm2(x)

        x = residual + self.dropout(
            self.src_attn(x, memory, memory, memory_mask))
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
            normalize_before: bool = True,
            concat_after: bool = False,
            compute_type=mstype.float32,
    ):
        super().__init__()
        attention_dim = encoder_output_size
        self.first_flag = True
        self.normalize_before = normalize_before

        self.output_layer = Dense(attention_dim,
                                  vocab_size).to_float(compute_type)

        self.embed = nn.SequentialCell(
            nn.Embedding(vocab_size, attention_dim),
            PositionalEncoding(attention_dim, positional_dropout_rate),
        )

        self.decoders = nn.CellList([
            TransformerDecoderLayer(
                attention_dim,
                attention_heads,
                self_attention_dropout_rate,
                dropout_rate,
                linear_units,
                normalize_before,
                concat_after,
                compute_type,
            ) for _ in range(num_blocks)
        ])

        if normalize_before:
            self.after_norm = LayerNorm(attention_dim, epsilon=1e-12)


        # self.expand_dims = ops.ExpandDims()
        # self.log_softmax = nn.LogSoftmax()

    def construct(
            self,
            memory: mindspore.Tensor,
            memory_mask: mindspore.Tensor,
            ys_in_pad: mindspore.Tensor,
            ys_masks: mindspore.Tensor,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_masks: mask for token sequences
        Returns:
            x: decoded token score before softmax (batch, maxlen_out,
                vocab_size) if use_output_layer is True
        """
        x, _ = self.embed(ys_in_pad)
        for layer in self.decoders:
            x, ys_masks, memory, memory_mask = layer(x, ys_masks, memory,
                                                     memory_mask)

        if self.normalize_before:
            x = self.after_norm(x)

        x = self.output_layer(x)

        return x


# if __name__ == '__main__':
#     import mindspore as ms
#     from mindspore import Tensor
#     from mindspore import dtype as mstype
#     import numpy as np
#
#     ms.set_context(mode=ms.PYNATIVE_MODE)
#     x = Tensor(np.ones([8, 512, 512]), mstype.float32)
#     mask = Tensor(np.ones([8, 512, 512]), mstype.float16)
#     mask_pad = Tensor(np.ones([8, 512, 512]), mstype.int32)
#     pos_embs = Tensor(np.ones([1, 512, 512]), mstype.float32)
#     net = ConformerEncoderLayer(size=x.shape, output_size=512, attention_heads=8, linear_units=512, cnn_module_kernel=3)
#     output, other = net(x, mask=mask, pos_emb=pos_embs, mask_pad=mask_pad)
