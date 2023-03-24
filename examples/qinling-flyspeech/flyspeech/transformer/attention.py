# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
# 2022.07 - Modified the code to support Mindspore
#           Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-Head Attention layer definition."""

import math
from typing import Optional, Tuple

import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from flyspeech.layers.dense import Dense
from mindspore.common.initializer import XavierUniform, initializer
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor


class MultiHeadedAttention(nn.Cell):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self, n_head: int, n_feat: int, dropout_rate: float, compute_type=mstype.float32
    ):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.neg_inf = Tensor([-10000.0], dtype=compute_type)
        self.scores_mul = Tensor([1.0 / math.sqrt(float(self.d_k))], dtype=compute_type)

        self.linear_q = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_k = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_v = Dense(n_feat, n_feat).to_float(compute_type)
        self.linear_out = Dense(n_feat, n_feat).to_float(compute_type)
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.softmax = nn.Softmax()

        self.expand_dims = ops.ExpandDims()
        self.equal = ops.Equal()
        self.matmul = ops.BatchMatMul()
        self.cast = ops.Cast()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.get_dtype = ops.DType()

    def forward_qkv(
        self, query: mindspore.Tensor, key: mindspore.Tensor, value: mindspore.Tensor
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """Transform query, key and value.

        Args:
            query (mindspore.Tensor): Query tensor (#batch, time1, size).
            key (mindspore.Tensor): Key tensor (#batch, time2, size).
            value (mindspore.Tensor): Value tensor (#batch, time2, size).

        Returns:
            mindspore.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            mindspore.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            mindspore.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).
        """
        n_batch = query.shape[0]
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(0, 2, 1, 3)  # (batch, head, time1, d_k)
        k = k.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)
        v = v.transpose(0, 2, 1, 3)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self,
        value: mindspore.Tensor,
        scores: mindspore.Tensor,
        mask: Optional[mindspore.Tensor],
    ) -> mindspore.Tensor:
        """Compute attention context vector.

        Args:
            value (mindspore.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (mindspore.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (mindspore.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            mindspore.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = value.shape[0]

        if mask is not None:
            attn_mask = self.expand_dims(mask, 1)
            attn_mask = self.cast(self.equal(attn_mask, 0), self.get_dtype(scores))
            if len(attn_mask.shape) == 3:
                attn_mask = self.expand_dims(attn_mask, 1)
            attn_mask = self.mul(attn_mask, self.neg_inf)
            scores = self.add(attn_mask, scores)
            attn = self.softmax(scores)  # (batch, head, time1, time2)
        else:
            attn = self.softmax(scores)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        x = self.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(0, 2, 1, 3).view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)  # (batch, time1, d_model)

    def construct(
        self,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        value: mindspore.Tensor,
        mask: Optional[mindspore.Tensor],
        pos_emb: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:  # pylint: disable=W0613
        """Compute scaled dot product attention.

        Args:
            query (mindspore.Tensor): Query tensor (#batch, time1, size).
            key (mindspore.Tensor): Key tensor (#batch, time2, size).
            value (mindspore.Tensor): Value tensor (#batch, time2, size).
            mask (mindspore.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            pos_emb (mindspore.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            mindspore.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = self.matmul(
            q * self.scores_mul, k.transpose(0, 1, 3, 2) * self.scores_mul
        )

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.

    Paper: https://arxiv.org/abs/1901.02860/
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate, compute_type):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, compute_type)
        # linear transformation for positional embeddings
        self.linear_pos = Dense(n_feat, n_feat, has_bias=False).to_float(compute_type)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = Parameter(
            initializer(XavierUniform(), [self.h, self.d_k], mstype.float32)
        )
        self.pos_bias_v = Parameter(
            initializer(XavierUniform(), [self.h, self.d_k], mstype.float32)
        )
        self.tile = ops.Tile()
        self.norm_factor = Tensor(1.0 / math.sqrt(self.d_k), mindspore.float16)

    def construct(
        self,
        query: mindspore.Tensor,
        key: mindspore.Tensor,
        value: mindspore.Tensor,
        mask: Optional[mindspore.Tensor],
        pos_emb: Optional[mindspore.Tensor] = None,
    ):
        """Compute 'Scaled Dot Product Attention' with rel.

        positional encoding.
        Args:
            query (mindspore.Tensor): Query tensor (#batch, time1, size).
            key (mindspore.Tensor): Key tensor (#batch, time2, size).
            value (mindspore.Tensor): Value tensor (#batch, time2, size).
            mask (mindspore.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            pos_emb (mindspore.Tensor): Positional embedding tensor
                (#batch, time2, size).
        Returns:
            mindspore.Tensor: Output tensor (#batch, time1, d_model).
        """
        n_batch = query.shape[0]
        n_batch_pos = pos_emb.shape[0]

        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(0, 2, 1, 3)  # (batch, time1, head, d_k)

        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(0, 2, 1, 3)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.cast(self.pos_bias_u, self.get_dtype(q))).transpose(
            0, 2, 1, 3
        )
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.cast(self.pos_bias_v, self.get_dtype(q))).transpose(
            0, 2, 1, 3
        )

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = self.matmul(q_with_bias_u, k.transpose(0, 1, 3, 2))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        p = self.tile(p, (n_batch, 1, 1, 1))
        matrix_bd = self.matmul(q_with_bias_v, p.transpose(0, 1, 3, 2))
        # Remove relative shift of matrix_bd since it is useless in speech recognition,
        # and it requires special attention for streaming.
        scores = matrix_ac + matrix_bd
        scores = self.mul(scores, self.scores_mul)

        return self.forward_attention(v, scores, mask)
