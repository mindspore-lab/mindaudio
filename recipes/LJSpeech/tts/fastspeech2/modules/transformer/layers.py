from mindspore import nn

from modules.transformer.sublayers import MultiHeadAttention
from modules.transformer.sublayers import PositionwiseFeedForward


class FFTBlock(nn.Cell):
    def __init__(
        self,
        d_model,
        d_inner,
        kernel_size,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
    ):
        super().__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, kernel_size, dropout=dropout)

    def construct(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output
