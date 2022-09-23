''' basic rnn cells '''
import math

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform


def rnn_tanh_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''rnn tanh cell '''
    if b_ih is None:
        igates = ops.MatMul(False, True)(inputs, w_ih)
        hgates = ops.MatMul(False, True)(hidden, w_hh)
    else:
        igates = ops.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = ops.MatMul(False, True)(hidden, w_hh) + b_hh
    return ops.Tanh()(igates + hgates)


def rnn_relu_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''rnn relu cell '''
    if b_ih is None:
        igates = ops.MatMul(False, True)(inputs, w_ih)
        hgates = ops.MatMul(False, True)(hidden, w_hh)
    else:
        igates = ops.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = ops.MatMul(False, True)(hidden, w_hh) + b_hh
    return ops.ReLU()(igates + hgates)


class LSTMCell(nn.Cell):
    '''lstm cell '''
    def __init__(self):
        super(LSTMCell, self).__init__()
        self.matmul = ops.MatMul(False, True)
        self.split = ops.Split(1, 4)
        self.cast = ops.Cast()
        self.tanh = ops.Tanh()
        self.sigmoid = ops.Sigmoid()

    def construct(self, inputs, hidden, w_ih, w_hh, b_ih, b_hh):
        ''' lstm '''
        hx, cx = hidden
        inputs = self.cast(inputs, mindspore.float16)
        hx = self.cast(hx, mindspore.float16)
        cx = self.cast(cx, mindspore.float16)
        w_ih = self.cast(w_ih, mindspore.float16)
        w_hh = self.cast(w_hh, mindspore.float16)
        b_ih = self.cast(b_ih, mindspore.float16)
        b_hh = self.cast(b_hh, mindspore.float16)
        if b_ih is None:
            gates = self.matmul(inputs, w_ih) + self.matmul(hx, w_hh)
        else:
            gates = self.matmul(inputs, w_ih) + \
                self.matmul(hx, w_hh) + b_ih + b_hh
        gates = self.cast(gates, mindspore.float32)
        ingate, forgetgate, cellgate, outgate = self.split(gates)

        ingate = self.sigmoid(ingate)
        forgetgate = self.sigmoid(forgetgate)
        cellgate = self.tanh(cellgate)
        outgate = self.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.tanh(cy)
        return hy, cy


def lstm_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    ''' lstm cell '''
    hx, cx = hidden

    if b_ih is None:
        gates = ops.MatMul(False, True)(inputs, w_ih) + \
            ops.MatMul(False, True)(hx, w_hh)
    else:
        gates = ops.MatMul(False, True)(inputs, w_ih) + \
            ops.MatMul(False, True)(hx, w_hh) + b_ih + b_hh

    ingate, forgetgate, cellgate, outgate = ops.Split(1, 4)(gates)

    ingate = ops.Sigmoid()(ingate)
    forgetgate = ops.Sigmoid()(forgetgate)
    cellgate = ops.Tanh()(cellgate)
    outgate = ops.Sigmoid()(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * ops.Tanh()(cy)

    return hy, cy


def gru_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    ''' gru cell '''
    if b_ih is None:
        gi = ops.MatMul(False, True)(inputs, w_ih)
        gh = ops.MatMul(False, True)(hidden, w_hh)
    else:
        gi = ops.MatMul(False, True)(inputs, w_ih) + b_ih
        gh = ops.MatMul(False, True)(hidden, w_hh) + b_hh
    i_r, i_i, i_n = ops.Split(1, 3)(gi)
    h_r, h_i, h_n = ops.Split(1, 3)(gh)

    resetgate = ops.Sigmoid()(i_r + h_r)
    inputgate = ops.Sigmoid()(i_i + h_i)
    newgate = ops.Tanh()(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


class RNNCellBase(nn.Cell):
    ''' rnn cell base '''
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            num_chunks: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(
            Tensor(
                np.random.randn(
                    num_chunks *
                    hidden_size,
                    input_size).astype(
                        np.float32)))
        self.weight_hh = Parameter(
            Tensor(
                np.random.randn(
                    num_chunks *
                    hidden_size,
                    hidden_size).astype(
                        np.float32)))
        if bias:
            self.bias_ih = Parameter(
                Tensor(
                    np.random.randn(
                        num_chunks *
                        hidden_size).astype(
                            np.float32)))
            self.bias_hh = Parameter(
                Tensor(
                    np.random.randn(
                        num_chunks *
                        hidden_size).astype(
                            np.float32)))
        self.reset_parameters()

    def reset_parameters(self):
        ''' init '''
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))


class RNNCell(RNNCellBase):
    ''' rnn cell '''
    _non_linearity = ['tanh', 'relu']

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True,
            nonlinearity: str = "tanh"):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        if nonlinearity not in self._non_linearity:
            raise ValueError(
                "Unknown nonlinearity: {}".format(
                    nonlinearity))
        self.nonlinearity = nonlinearity

    def construct(self, inputs, hx):
        ''' rnn cell '''
        if self.nonlinearity == "tanh":
            ret = rnn_tanh_cell(
                inputs,
                hx,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh)
        else:
            ret = rnn_relu_cell(
                inputs,
                hx,
                self.weight_ih,
                self.weight_hh,
                self.bias_ih,
                self.bias_hh)
        return ret
