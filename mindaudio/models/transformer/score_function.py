import mindspore.numpy as msnp
from mindspore import nn, ops


class ScaledDotProductAttention(nn.Cell):
    """
    Scaled Dot-Product Attention.
    """

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature

        self.softmax = nn.Softmax(axis=2)
        self.dropout = nn.Dropout(p=attn_dropout)

        self.bmm = ops.BatchMatMul()
        self.transpose = ops.Transpose()
        self.zeros_like = ops.ZerosLike()

    def construct(self, q, k, v, mask=None):
        attn = self.bmm(q, self.transpose(k, (0, 2, 1)))
        attn = attn / self.temperature

        inf_mask = self.zeros_like(attn) - msnp.inf

        if mask is not None:
            attn = msnp.where(mask, inf_mask, attn)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = self.bmm(attn, v)

        return output
