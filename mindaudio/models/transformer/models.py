
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import ops

from mindaudio.models.transformer import constants
from mindaudio.models.transformer.layers import FFTBlock
from mindaudio.models.transformer.positional_encoding import get_sinusoid_encoding_table


class Encoder(nn.Cell):
    def __init__(self, hps):
        super().__init__()

        n_src_vocab = hps.model.n_src_vocab + 1
        len_max_seq = hps.model.max_seq_len
        d_word_vec = hps.model.transformer.encoder_hidden
        n_layers = hps.model.transformer.encoder_layer
        n_head = hps.model.transformer.encoder_head
        d_k = hps.model.transformer.encoder_hidden // hps.model.transformer.encoder_head
        d_v = hps.model.transformer.encoder_hidden // hps.model.transformer.encoder_head
        d_model = hps.model.transformer.encoder_hidden
        d_inner = hps.model.transformer.conv_filter_size
        kernel_size = hps.model.transformer.conv_kernel_size
        dropout = hps.model.transformer.encoder_dropout

        n_position = len_max_seq + 1
        pretrained_embs = get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=None)
        self.position_enc = ops.stop_gradient(Tensor(pretrained_embs)[None, ...])

        self.src_word_emb = nn.Embedding(
            n_src_vocab,
            d_word_vec,
            padding_idx=constants.PAD,
        )

        self.layer_stack = nn.CellList(
            [
                FFTBlock(d_model, d_inner, kernel_size, n_head, d_k, d_v, dropout) for _ in range(n_layers)
            ]
        )

        self.n_head = n_head
        self.equal = ops.Equal()
        self.not_equal = ops.NotEqual()
        self.expand_dims = ops.ExpandDims()
        self.pad = constants.PAD

    def construct(self, src_seq, src_pos, positions_encoder=None):
        padding_mask = self.equal(src_seq, self.pad)
        slf_attn_mask = self.expand_dims(padding_mask.astype(mstype.float32), 1)
        slf_attn_mask = ops.tile(slf_attn_mask, (self.n_head, 1, 1))
        slf_attn_mask_bool = slf_attn_mask.astype(mstype.bool_)

        non_pad_mask_bool = self.expand_dims(self.not_equal(src_seq, self.pad), 2)
        non_pad_mask = non_pad_mask_bool.astype(mstype.float32)
        if positions_encoder is not None:
            enc_output = self.src_word_emb(src_seq.astype('int32')) + positions_encoder
        else:
            max_len = src_seq.shape[1]
            enc_output = self.src_word_emb(src_seq.astype('int32')) + self.position_enc[:, : max_len]
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_bool,
            )
        return enc_output


class Decoder(nn.Cell):
    def __init__(self, hps):
        super().__init__()
        n_position = hps.model.max_seq_len + 1
        n_layers = hps.model.transformer.decoder_layer
        n_head = hps.model.transformer.decoder_head
        d_k = hps.model.transformer.decoder_hidden // hps.model.transformer.decoder_head
        d_v = hps.model.transformer.decoder_hidden // hps.model.transformer.decoder_head
        d_model = hps.model.transformer.decoder_hidden
        d_inner = hps.model.transformer.conv_filter_size
        kernel_size = hps.model.transformer.conv_kernel_size
        dropout = hps.model.transformer.decoder_dropout

        len_max_seq = hps.model.max_seq_len
        self.max_seq_len = hps.model.max_seq_len

        n_position = len_max_seq + 1
        pretrained_embs = get_sinusoid_encoding_table(n_position, d_model, padding_idx=None)
        self.position_enc = ops.stop_gradient(Tensor(pretrained_embs)[None, ...])

        self.layer_stack = nn.CellList(
            [
                FFTBlock(d_model, d_inner, kernel_size, n_head, d_k, d_v, dropout) for _ in range(n_layers)
            ]
        )
        self.n_head = n_head
        self.pad = constants.PAD
        self.equal = ops.Equal()
        self.not_equal = ops.NotEqual()
        self.expand_dims = ops.ExpandDims()

    def construct(self, enc_seq, mask, positions_decoder=None):
        slf_attn_mask = self.expand_dims(mask.astype(mstype.float32), 1)
        slf_attn_mask_bool = slf_attn_mask.astype(mstype.bool_)
        slf_attn_mask_bool_tile = ops.tile(slf_attn_mask_bool, (self.n_head, 1, 1))

        non_pad_mask = 1. - mask.expand_dims(2)

        if positions_decoder is not None:
            dec_output = enc_seq + positions_decoder
        else:
            batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
            dec_output = enc_seq + self.position_enc[:, :max_len, :]

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_bool_tile)

        return dec_output, slf_attn_mask_bool
