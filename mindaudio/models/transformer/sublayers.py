import numpy as np
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import Normal, initializer

from mindaudio.models.transformer.score_function import ScaledDotProductAttention


class MultiHeadAttention(nn.Cell):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Dense(
            d_model,
            n_head * d_k,
            weight_init=initializer(
                Normal(sigma=np.sqrt(2.0 / (d_model + d_k)), mean=0),
                [d_model, n_head * d_k],
                mstype.float32,
            ),
        )

        self.w_ks = nn.Dense(
            d_model,
            n_head * d_k,
            weight_init=initializer(
                Normal(sigma=np.sqrt(2.0 / (d_model + d_k)), mean=0),
                [d_model, n_head * d_k],
                mstype.float32,
            ),
        )

        self.w_vs = nn.Dense(
            d_model,
            n_head * d_v,
            weight_init=initializer(
                Normal(sigma=np.sqrt(2.0 / (d_model + d_v)), mean=0),
                [d_model, n_head * d_v],
                mstype.float32,
            ),
        )

        self.fc = nn.Dense(
            n_head * d_v,
            d_model,
            weight_init=initializer(Normal(), [n_head * d_v, d_model], mstype.float32),
        )

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5), attn_dropout=dropout
        )
        self.layer_norm = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(p=dropout)

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()

    def construct(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        residual = q

        q = self.reshape(self.w_qs(q), (sz_b, len_q, n_head, d_k))
        k = self.reshape(self.w_ks(k), (sz_b, len_k, n_head, d_k))
        v = self.reshape(self.w_vs(v), (sz_b, len_v, n_head, d_v))

        q = self.reshape(
            self.transpose(q, (2, 0, 1, 3)), (-1, len_q, d_k)
        )  # (n*b) x lq x dk
        k = self.reshape(
            self.transpose(k, (2, 0, 1, 3)), (-1, len_q, d_k)
        )  # (n*b) x lk x dk
        v = self.reshape(
            self.transpose(v, (2, 0, 1, 3)), (-1, len_v, d_v)
        )  # (n*b) x lv x dv

        output = self.attention(q, k, v, mask=mask)

        output = self.reshape(output, (n_head, sz_b, len_q, d_v))
        output = self.reshape(
            self.transpose(output, (1, 2, 0, 3)), (sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class PositionwiseFeedForward(nn.Cell):
    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Conv1d(
            d_in,
            d_hid,
            kernel_size=kernel_size[0],
            pad_mode="same",
            has_bias=True,
        )

        self.w_2 = nn.Conv1d(
            d_hid,
            d_in,
            kernel_size=kernel_size[1],
            pad_mode="same",
            has_bias=True,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm([d_in])
        self.relu = nn.ReLU()

        self.transpose = ops.Transpose()

    def construct(self, x):
        residual = x

        output = self.transpose(x, (0, 2, 1))
        output = self.w_1(output)
        output = self.relu(output)
        output = self.w_2(output)
        output = self.transpose(output, (0, 2, 1))
        output = self.dropout(output)

        output = self.layer_norm(output + residual)

        return output
