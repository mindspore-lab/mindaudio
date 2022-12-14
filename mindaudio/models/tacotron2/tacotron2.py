'''tacotron2 model '''
import math

import numpy as np
import mindspore
from mindspore import nn
import mindspore.ops as ops
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.parallel._utils import  _get_gradients_mean
from mindspore import Parameter, Tensor

from mindaudio.models.tacotron2.rnns import LSTM
from mindaudio.models.tacotron2.hparams import hparams as hps


gain = {'linear': 1, 'sigmoid': 1, 'tanh': 5 / 3, 'relu': math.sqrt(2)}


class LinearNorm(nn.Cell):
    '''linear layer'''
    def __init__(self, in_channels, out_channels, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()

        w_init = mindspore.common.initializer.XavierUniform(gain=gain[w_init_gain])
        self.linear_layer = nn.Dense(
            in_channels,
            out_channels,
            has_bias=bias,
            weight_init=w_init
        ).to_float(mindspore.float16)

        self.cast = ops.Cast()

    def construct(self, x):
        ''' construct '''
        return self.cast(self.linear_layer(x), mindspore.float32)


class ConvNorm(nn.Cell):
    '''conv1d layer'''
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain='linear'
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            padding = int(dilation * (kernel_size - 1) / 2)

        w_init = mindspore.common.initializer.XavierUniform(gain=gain[w_init_gain])
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode='pad',
            padding=padding,
            dilation=dilation,
            has_bias=bias,
            weight_init=w_init
        )

    def construct(self, signal):
        return self.conv(signal)


class Tacotron2Loss(nn.Cell):
    ''' tacotron loss calculate '''
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.reshape = ops.Reshape()
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

        self.n_frames_per_step = hps.n_frames_per_step
        self.p = hps.p

    def construct(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_out, mel_out_postnet, gate_out, _ = model_output

        gate_target = self.reshape(
            gate_target[:, ::self.n_frames_per_step], (-1, 1))
        gate_out = self.reshape(gate_out, (-1, 1))
        mel_loss = self.mse(self.p * mel_out, self.p * mel_target) + \
            self.mse(self.p * mel_out_postnet, self.p * mel_target)
        gate_loss = self.bce(gate_out, gate_target)
        print('mel_loss:', mel_loss)
        print('gate_loss:', gate_loss)
        return mel_loss + gate_loss


class LocationLayer(nn.Cell):
    ''' location layer '''
    def __init__(self, 
        attention_n_filters,
        attention_kernel_size,
        attention_dim
    ):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1
        )
        self.location_dense = LinearNorm(
            attention_n_filters,
            attention_dim,
            bias=False,
            w_init_gain='tanh')
        self.transpose = ops.Transpose()

    def construct(self, attention_weights_cat):
        ''' construct '''
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = self.transpose(processed_attention, (0, 2, 1))
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Cell):
    '''attention layer '''
    def __init__(
            self,
            memory_layer,
            attention_rnn_dim,
            embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = memory_layer

        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim)

        self.expand_dims = ops.ExpandDims()
        self.tanh = ops.Tanh()
        self.reshape = ops.Reshape()
        self.squeeze = ops.Squeeze(-1)
        self.softmax = ops.Softmax(-1)
        self.bmm = ops.BatchMatMul()
        self.squeeze_ = ops.Squeeze(1)
        self.select = ops.Select()
        self.fill = ops.Fill()
        self.get_shape = ops.Shape()
        self.score_values = -float('inf')

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        '''get alignment '''
        processed_query = self.expand_dims(self.query_layer(query), 1)

        processed_attention_weights = self.location_layer(
            attention_weights_cat)
        processed_attention = self.tanh(
            processed_query +
            processed_attention_weights +
            processed_memory)
        energies = self.v(processed_attention)
        energies = self.squeeze(energies)
        return energies

    def construct(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat,
            mask=None):
        ''' construct '''
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment = self.select(
                mask,
                alignment,
                self.fill(
                    mindspore.float32,
                    self.get_shape(mask),
                    self.score_values))

        attention_weights = self.softmax(alignment)
        attention_context = self.bmm(
            self.expand_dims(
                attention_weights, 1), memory)
        attention_context = self.squeeze_(attention_context)
        return attention_context, attention_weights

    def inference(
            self,
            attention_hidden_state,
            memory,
            processed_memory,
            attention_weights_cat,
            mask=None):
        ''' construct '''
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        attention_weights = self.softmax(alignment)
        attention_context = self.bmm(
            self.expand_dims(
                attention_weights, 1), memory)
        attention_context = self.squeeze_(attention_context)
        return attention_context, attention_weights

class Prenet(nn.Cell):
    ''' prenet '''
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        layers = [nn.SequentialCell([LinearNorm(in_size, out_size, bias=False)])
                  for (in_size, out_size) in zip(in_sizes, sizes)]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.size = sizes[-1]
        self.layers = nn.CellList(layers)

    def construct(self, x):
        ''' construct '''
        for linear in self.layers:
            x = self.dropout(self.relu(linear(x)))
        return x


class Postnet(nn.Cell):
    ''' postnet '''
    def __init__(self):
        super(Postnet, self).__init__()
        conv_layer = []
        conv_layer.extend(nn.SequentialCell([
            ConvNorm(hps.num_mels, hps.postnet_embedding_dim,
                     kernel_size=hps.postnet_kernel_size, stride=1,
                     padding=int((hps.postnet_kernel_size - 1) / 2),
                     dilation=1, w_init_gain='tanh'),
            ExpandDims(),
            nn.BatchNorm2d(hps.postnet_embedding_dim),
            Squeeze(),
            nn.Tanh(),
            nn.Dropout(keep_prob=0.5)
        ]))

        for _ in range(1, hps.postnet_n_convolutions - 1):
            conv_layer.extend(nn.SequentialCell([
                ConvNorm(hps.postnet_embedding_dim,
                         hps.postnet_embedding_dim,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                ExpandDims(),
                nn.BatchNorm2d(hps.postnet_embedding_dim),
                Squeeze(),
                nn.Tanh(),
                nn.Dropout(keep_prob=0.5)]))

        conv_layer.extend(
            nn.SequentialCell(
                [
                    ConvNorm(
                        hps.postnet_embedding_dim,
                        hps.num_mels,
                        kernel_size=hps.postnet_kernel_size,
                        stride=1,
                        padding=int(
                            (hps.postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain='linear'),
                    ExpandDims(),
                    nn.BatchNorm2d(
                        hps.num_mels),
                    Squeeze(),
                    nn.Dropout(
                        keep_prob=0.5)]))
        self.convolutions = nn.CellList(conv_layer)

    def construct(self, x):
        ''' construct '''
        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
        return x

    def inference(self, x):
        '''inference '''
        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
        return x


class ExpandDims(nn.Cell):
    '''expand dim'''
    def __init__(self):
        super(ExpandDims, self).__init__()
        self.expand_dim = ops.ExpandDims()

    def construct(self, x):
        ''' construct '''
        return self.expand_dim(x, -1)


class Squeeze(nn.Cell):
    ''' squeeze dim '''
    def __init__(self):
        super(Squeeze, self).__init__()
        self.squeeze = ops.Squeeze(-1)

    def construct(self, x):
        ''' construct '''
        return self.squeeze(x)


class Encoder(nn.Cell):
    ''' encoder '''
    def __init__(self):
        super(Encoder, self).__init__()

        conv_layer = []
        for _ in range(hps.encoder_n_convolutions):
            conv_layer.extend(nn.SequentialCell([
                ConvNorm(hps.encoder_embedding_dim,
                         hps.encoder_embedding_dim,
                         kernel_size=hps.encoder_kernel_size, stride=1,
                         padding=int((hps.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                ExpandDims(),
                nn.BatchNorm2d(hps.encoder_embedding_dim),
                Squeeze(),
                nn.ReLU(),
                nn.Dropout(keep_prob=0.5)]))

        self.convolutions = nn.CellList(conv_layer)

        self.lstm = LSTM(
            input_size=hps.encoder_embedding_dim,
            hidden_size=int(
                hps.encoder_embedding_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.transpose = ops.Transpose()
        self.cast = ops.Cast()
        self.h, self.c = self.lstm_default_state(hps.batch_size, int(
            hps.encoder_embedding_dim / 2), 1, bidirectional=True)
        self.h_test, self.c_test = self.lstm_default_state(
            hps.test_batch_size, int(
                hps.encoder_embedding_dim / 2), 1, bidirectional=True)
        self.fullzeros = Tensor(
            np.zeros(
                (hps.batch_size,
                 hps.max_text_len,
                 512)),
            mindspore.float32)
        self.select = ops.Select()

    def lstm_default_state(
            self,
            batch_size,
            hidden_size,
            num_layers,
            bidirectional):
        ''' init lstm '''
        num_directions = 2 if bidirectional else 1
        h = Tensor(
            np.zeros(
                (num_layers *
                 num_directions,
                 batch_size,
                 hidden_size)),
            mindspore.float32)
        c = Tensor(
            np.zeros(
                (num_layers *
                 num_directions,
                 batch_size,
                 hidden_size)),
            mindspore.float32)
        return h, c

    def construct(self, x, input_length, mask):
        ''' construct '''
        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
        x = self.transpose(x, (0, 2, 1))

        outputs, _ = self.lstm(x, h=(self.h, self.c), seq_length=input_length)
        outputs = mask * outputs

        outputs = self.cast(outputs, mindspore.float32)
        return outputs

    def inference(self, x):
        '''inference '''
        for layer in self.convolutions:
            x = layer(x)
        x = self.transpose(x, (0, 2, 1))
        outputs, _ = self.lstm(x, h=(self.h_test, self.c_test))

        outputs = self.cast(outputs, mindspore.float32)

        return outputs


class LSTMCell(nn.Cell):
    '''lstm cell '''
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        w_init = mindspore.common.initializer.Uniform(
            scale=1 / math.sqrt(hidden_size))
        self.linear1 = nn.Dense(
            input_size,
            4 * hidden_size,
            weight_init=w_init).to_float(
                mindspore.float16)
        self.linear2 = nn.Dense(
            hidden_size,
            4 * hidden_size,
            weight_init=w_init).to_float(
                mindspore.float16)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.split = ops.Split(1, 4)
        self.cast = ops.Cast()

    def construct(self, inputs, hx, cx):
        ''' construct '''
        gates = self.cast(self.linear2(hx), mindspore.float32) + \
            self.cast(self.linear1(inputs), mindspore.float32)
        ingate, forgetgate, cellgate, outgate = self.split(gates)
        ingate = self.sigmoid(ingate)
        forgetgate = self.sigmoid(forgetgate)
        cellgate = self.tanh(cellgate)
        outgate = self.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.tanh(cy)
        return hy, cy


class Decode(nn.Cell):
    ''' decode at each step '''
    def __init__(self, memory_layer):
        super(Decode, self).__init__()
        self.num_mels = hps.num_mels
        self.n_frames_per_step = hps.n_frames_per_step
        self.encoder_embedding_dim = hps.encoder_embedding_dim
        self.attention_rnn_dim = hps.attention_rnn_dim
        self.decoder_rnn_dim = hps.decoder_rnn_dim
        self.prenet_dim = hps.prenet_dim
        self.max_decoder_steps = hps.max_decoder_steps
        self.gate_threshold = hps.gate_threshold
        self.p_attention_dropout = hps.p_attention_dropout
        self.p_decoder_dropout = hps.p_decoder_dropout

        self.attention_rnn = LSTMCell(
            hps.prenet_dim + hps.encoder_embedding_dim,
            hps.attention_rnn_dim)

        self.attention_layer = Attention(
            memory_layer,
            hps.attention_rnn_dim, hps.encoder_embedding_dim,
            hps.attention_dim, hps.attention_location_n_filters,
            hps.attention_location_kernel_size)

        self.decoder_rnn = LSTMCell(
            hps.attention_rnn_dim + hps.encoder_embedding_dim,
            hps.decoder_rnn_dim)

        self.linear_projection = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim,
            hps.num_mels * hps.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

        self.dropout_attention = nn.Dropout(
            keep_prob=1 - self.p_attention_dropout)
        self.dropout_decoder = nn.Dropout(keep_prob=1 - self.p_decoder_dropout)

        self.concat_ = ops.Concat(-1)
        self.concat_dim1 = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()
        self.squeeze_dim1 = ops.Squeeze(1)

    def construct(self, decoder_input, attention_hidden,
                  attention_cell, attention_weights, attention_weights_cum,
                  attention_context, memory, processed_memory,
                  decoder_hidden, decoder_cell, mask):
        ''' construct '''
        cell_input = self.concat_((decoder_input, attention_context))
        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, attention_hidden, attention_cell)

        attention_hidden = self.dropout_attention(attention_hidden)

        attention_weights_cat = self.concat_dim1(
            (self.expand_dims(attention_weights, 1),
             self.expand_dims(attention_weights_cum, 1)))

        attention_context, attention_weights = self.attention_layer(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        attention_weights_cum += attention_weights
        decoder_input = self.concat_(
            (attention_hidden, attention_context))

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, decoder_hidden, decoder_cell)

        decoder_hidden = self.dropout_decoder(decoder_hidden)

        decoder_hidden_attention_context = self.concat_dim1(
            (decoder_hidden, attention_context))

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (
            decoder_output,
            gate_prediction,
            attention_weights,
            attention_weights_cum,
            attention_context,
            decoder_hidden,
            decoder_cell,
            attention_hidden,
            attention_cell)

    def inference(self, decoder_input, attention_hidden,
                  attention_cell, attention_weights, attention_weights_cum,
                  attention_context, memory, processed_memory,
                  decoder_hidden, decoder_cell, mask):
        ''' construct '''
        cell_input = self.concat_((decoder_input, attention_context))
        attention_hidden, attention_cell = self.attention_rnn(
            cell_input, attention_hidden, attention_cell)

        attention_hidden = self.dropout_attention(attention_hidden)

        attention_weights_cat = self.concat_dim1(
            (self.expand_dims(attention_weights, 1),
             self.expand_dims(attention_weights_cum, 1)))

        attention_context, attention_weights = self.attention_layer.inference(
            attention_hidden, memory, processed_memory,
            attention_weights_cat, mask)

        attention_weights_cum += attention_weights
        decoder_input = self.concat_(
            (attention_hidden, attention_context))

        decoder_hidden, decoder_cell = self.decoder_rnn(
            decoder_input, decoder_hidden, decoder_cell)

        decoder_hidden = self.dropout_decoder(decoder_hidden)

        decoder_hidden_attention_context = self.concat_dim1(
            (decoder_hidden, attention_context))

        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return (
            decoder_output,
            gate_prediction,
            attention_weights,
            attention_weights_cum,
            attention_context,
            decoder_hidden,
            decoder_cell,
            attention_hidden,
            attention_cell)

class Decoder(nn.Cell):
    ''' decoder '''
    def __init__(self):
        super(Decoder, self).__init__()

        self.num_mels = hps.num_mels
        self.n_frames_per_step = hps.n_frames_per_step
        self.encoder_embedding_dim = hps.encoder_embedding_dim
        self.attention_rnn_dim = hps.attention_rnn_dim
        self.decoder_rnn_dim = hps.decoder_rnn_dim
        self.prenet_dim = hps.prenet_dim
        self.max_decoder_steps = hps.max_decoder_steps
        self.gate_threshold = hps.gate_threshold
        self.p_attention_dropout = hps.p_attention_dropout
        self.p_decoder_dropout = hps.p_decoder_dropout

        self.memory_layer = LinearNorm(
            hps.encoder_embedding_dim,
            hps.attention_dim,
            bias=False,
            w_init_gain='tanh')

        self.prenet = Prenet(
            hps.num_mels * hps.n_frames_per_step,
            [hps.prenet_dim, hps.prenet_dim])
        self.reshape = ops.Reshape()
        self.get_shape = ops.Shape()
        self.transpose = ops.Transpose()
        self.concat = ops.Concat()
        self.concat_ = ops.Concat(-1)
        self.concat_dim1 = ops.Concat(axis=1)
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()
        self.squeeze_dim1 = ops.Squeeze(1)
        self.fill = ops.Fill()
        self.zeros = ops.Zeros()
        self.pack = ops.Stack()
        self.decode = Decode(self.memory_layer)
        self.sigmoid = ops.Sigmoid()
        self.concat_len = 50
        self.attention_zero_tensor = Tensor(
            np.zeros((hps.batch_size, self.attention_rnn_dim)), mindspore.float32)
        self.decoder_zero_tensor = Tensor(
            np.zeros((hps.batch_size, self.decoder_rnn_dim)), mindspore.float32)
        self.attention_context = Tensor(
            np.zeros((hps.batch_size, self.encoder_embedding_dim)), mindspore.float32)
        self.go_frame = Tensor(
            np.zeros(
                (hps.batch_size,
                 self.num_mels *
                 self.n_frames_per_step)),
            mindspore.float32)

    def parse_decoder_inputs(self, decoder_inputs):
        ''' parse decoder inputs '''
        decoder_inputs = self.transpose(decoder_inputs, (0, 2, 1))

        B, n_frame, _ = self.get_shape(decoder_inputs)

        decoder_inputs = self.reshape(
            decoder_inputs, (B, n_frame // self.n_frames_per_step, -1))
        decoder_inputs = self.transpose(decoder_inputs, (1, 0, 2))
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        ''' pack outputs '''
        align_tuple = ()
        n_frames = len(alignments)
        for i in range(n_frames // self.concat_len):
            start = i * self.concat_len
            end = (i + 1) * self.concat_len
            alignment = self.pack(alignments[start: end])
            align_tuple += (alignment,)
        if n_frames % self.concat_len != 0:
            start = n_frames // self.concat_len * self.concat_len
            alignment = self.pack(alignments[start:])
            align_tuple += (alignment,)
        alignments = self.concat(align_tuple)
        alignments = self.transpose(alignments, (1, 0, 2))

        gate_tuple = ()
        for i in range(n_frames // self.concat_len):
            start = i * self.concat_len
            end = (i + 1) * self.concat_len
            gate_output = self.pack(gate_outputs[start: end])
            gate_tuple += (gate_output,)
        if n_frames % self.concat_len != 0:
            start = n_frames // self.concat_len * self.concat_len
            gate_output = self.pack(gate_outputs[start:])
            gate_tuple += (gate_output,)

        gate_outputs = self.concat(gate_tuple)
        if len(self.get_shape(gate_outputs)) <= 1:
            gate_outputs = self.expand_dims(gate_outputs, 0)
        gate_outputs = self.transpose(gate_outputs, (1, 0))

        mel_tuple = ()
        for i in range(n_frames // self.concat_len):
            start = i * self.concat_len
            end = (i + 1) * self.concat_len
            mel_output = self.pack(mel_outputs[start: end])
            mel_tuple += (mel_output,)
        if n_frames % self.concat_len != 0:
            start = n_frames // self.concat_len * self.concat_len
            mel_output = self.pack(mel_outputs[start:])
            mel_tuple += (mel_output,)
        mel_outputs = self.concat(mel_tuple)
        mel_outputs = self.transpose(mel_outputs, (1, 0, 2))
        mel_outputs = self.reshape(
            mel_outputs, (self.get_shape(mel_outputs)[0], -1, self.num_mels))
        mel_outputs = self.transpose(mel_outputs, (0, 2, 1))

        return mel_outputs, gate_outputs, alignments

    def construct(self, memory, decoder_inputs, text_mask):
        ''' construct '''
        decoder_input = self.expand_dims(self.go_frame, 0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs_ = self.concat((decoder_input, decoder_inputs))
        decoder_inputs = self.prenet(decoder_inputs_)

        B, MAX_TIME, _ = self.get_shape(memory)

        attention_hidden = self.attention_zero_tensor
        attention_cell = self.attention_zero_tensor

        decoder_hidden = self.decoder_zero_tensor
        decoder_cell = self.decoder_zero_tensor

        attention_weights = self.fill(mindspore.float32, (B, MAX_TIME), 0.0)
        attention_weights_cum = self.fill(mindspore.float32, (B, MAX_TIME), 0.0)
        attention_context = self.attention_context

        processed_memory = self.memory_layer(memory)

        mask = text_mask

        mel_outputs, gate_outputs, alignments = (), (), ()

        n_frame, _, _ = self.get_shape(decoder_inputs)

        for i in range(n_frame - 1):
            decoder_input = self.squeeze(decoder_inputs[i:i + 1])
            mel_output, gate_output, attention_weights, attention_weights_cum, \
            attention_context, decoder_hidden, decoder_cell, attention_hidden, \
            attention_cell = self.decode(decoder_input,
                                         attention_hidden, attention_cell,
                                         attention_weights, attention_weights_cum, attention_context,
                                         memory, processed_memory,
                                         decoder_hidden, decoder_cell, mask)

            mel_outputs += (mel_output,)
            gate_outputs += (self.squeeze(gate_output),)
            alignments += (attention_weights,)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, text_mask):
        '''inference '''
        B, MAX_TIME, _ = self.get_shape(memory)

        decoder_input = self.fill(
            mindspore.float32, (B, self.num_mels * self.n_frames_per_step), 0)

        attention_hidden = self.zeros((B, self.attention_rnn_dim), mindspore.float32)
        attention_cell = self.zeros((B, self.attention_rnn_dim), mindspore.float32)

        decoder_hidden = self.zeros((B, self.decoder_rnn_dim), mindspore.float32)
        decoder_cell = self.zeros((B, self.decoder_rnn_dim), mindspore.float32)

        attention_weights = self.fill(mindspore.float32, (B, MAX_TIME), 0.0)
        attention_weights_cum = self.fill(mindspore.float32, (B, MAX_TIME), 0.0)
        attention_context = self.zeros(
            (B, self.encoder_embedding_dim), mindspore.float32)

        processed_memory = self.memory_layer(memory)

        mask = text_mask
        mel_outputs, gate_outputs, alignments = (), (), ()
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, attention_weights, attention_weights_cum, \
            attention_context, decoder_hidden, decoder_cell, attention_hidden, \
            attention_cell = self.decode.inference(decoder_input,
                                                   attention_hidden, attention_cell,
                                                   attention_weights, attention_weights_cum, attention_context,
                                                   memory, processed_memory,
                                                   decoder_hidden, decoder_cell, mask)

            mel_outputs += (mel_output,)
            gate_outputs += (self.squeeze(gate_output),)
            alignments += (attention_weights,)

            if self.sigmoid(gate_output[0]) > self.gate_threshold:
                ops.Print()('Terminated by gate.')
                break
            if len(mel_outputs) > 1 and (mel_output <= 0.2).all():
                ops.Print()('Warning: End with low power.')
                break
            if len(mel_outputs) == self.max_decoder_steps:
                ops.Print()('Warning: Reached max decoder steps.')
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inferencev2(self, memory, text_mask):
        '''inferencev2 '''
        B, MAX_TIME, _ = self.get_shape(memory)

        decoder_input = self.fill(
            mindspore.float32, (B, self.num_mels * self.n_frames_per_step), 0)

        attention_hidden = self.zeros((B, self.attention_rnn_dim), mindspore.float32)
        attention_cell = self.zeros((B, self.attention_rnn_dim), mindspore.float32)

        decoder_hidden = self.zeros((B, self.decoder_rnn_dim), mindspore.float32)
        decoder_cell = self.zeros((B, self.decoder_rnn_dim), mindspore.float32)

        attention_weights = self.fill(mindspore.float32, (B, MAX_TIME), 0.0)
        attention_weights_cum = self.fill(mindspore.float32, (B, MAX_TIME), 0.0)
        attention_context = self.zeros(
            (B, self.encoder_embedding_dim), mindspore.float32)

        processed_memory = self.memory_layer(memory)

        mask = text_mask
        mel_outputs, gate_outputs, alignments = (), (), ()
        for _ in range(292):
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, attention_weights, attention_weights_cum, \
            attention_context, decoder_hidden, decoder_cell, attention_hidden, \
            attention_cell = self.decode(decoder_input,
                                         attention_hidden, attention_cell,
                                         attention_weights, attention_weights_cum, attention_context,
                                         memory, processed_memory,
                                         decoder_hidden, decoder_cell, mask)

            mel_outputs += (mel_output,)
            gate_outputs += (self.squeeze(gate_output),)
            alignments += (attention_weights,)

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Cell):
    '''tacotron2 '''
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.num_mels = hps.num_mels
        self.mask_padding = hps.mask_padding
        self.n_frames_per_step = hps.n_frames_per_step

        std = math.sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = math.sqrt(3.0) * std
        w_init = mindspore.common.initializer.Uniform(scale=val)
        self.embedding = nn.Embedding(
            hps.n_symbols, hps.symbols_embedding_dim, embedding_table=w_init)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

        self.transpose = ops.Transpose()
        self.select = ops.Select()
        self.fill = ops.Fill()
        self.get_shape = ops.Shape()

    def parse_output(self, outputs, mel_mask=None):
        ''' parse output '''
        if mel_mask is not None:
            outputs[0] = self.select(
                mel_mask, outputs[0], self.fill(
                    mindspore.float32, self.get_shape(
                        outputs[0]), 0.0))
            outputs[1] = self.select(
                mel_mask, outputs[1], self.fill(
                    mindspore.float32, self.get_shape(
                        outputs[1]), 0.0))
            outputs[2] = self.select(mel_mask[:,
                                              0,
                                              ::self.n_frames_per_step],
                                     outputs[2],
                                     self.fill(mindspore.float32,
                                               self.get_shape(outputs[2]),
                                               1e3))

        return outputs

    def construct(
            self,
            text_inputs,
            input_length,
            mel_padded,
            text_mask,
            mel_mask,
            rnn_mask):
        ''' construct '''
        embedded_inputs = self.transpose(
            self.embedding(text_inputs), (0, 2, 1))

        encoder_outputs = self.encoder(embedded_inputs, input_length, rnn_mask)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_padded, text_mask)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            mel_mask)

    def inference(self, inputs, text_mask):
        '''inference '''
        embedded_inputs = self.transpose(self.embedding(inputs), (0, 2, 1))
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, text_mask)

        mel_outputs_postnet = self.postnet.inference(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs


class NetWithLossClass(nn.Cell):
    ''' net with loss'''
    def __init__(self, model, loss_fn):
        super(NetWithLossClass, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def construct(
            self,
            text_padded,
            input_length,
            mel_padded,
            gate_padded,
            text_mask,
            mel_mask,
            rnn_mask):
        ''' construct '''
        out = self.model(
            text_padded,
            input_length,
            mel_padded,
            text_mask,
            mel_mask,
            rnn_mask)
        loss = self.loss_fn(out, (mel_padded, gate_padded))
        return loss


class PredictMel(nn.Cell):
    '''predict cell for inference '''
    def __init__(self):
        super(PredictMel, self).__init__()
        self.num_mels = hps.num_mels
        self.mask_padding = hps.mask_padding
        self.n_frames_per_step = hps.n_frames_per_step

        std = math.sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = math.sqrt(3.0) * std
        w_init = mindspore.common.initializer.Uniform(scale=val)
        self.embedding = nn.Embedding(
            hps.n_symbols, hps.symbols_embedding_dim, embedding_table=w_init)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

        self.transpose = ops.Transpose()
        self.select = ops.Select()
        self.fill = ops.Fill()
        self.get_shape = ops.Shape()

    def parse_output(self, outputs, mel_mask=None):
        ''' parse output '''
        if mel_mask is not None:
            outputs[0] = self.select(
                mel_mask, outputs[0], self.fill(
                    mindspore.float32, self.get_shape(
                        outputs[0]), 0.0))
            outputs[1] = self.select(
                mel_mask, outputs[1], self.fill(
                    mindspore.float32, self.get_shape(
                        outputs[1]), 0.0))
            outputs[2] = self.select(mel_mask[:,
                                              0,
                                              ::self.n_frames_per_step],
                                     outputs[2],
                                     self.fill(mindspore.float32,
                                               self.get_shape(outputs[2]),
                                               1e3))

        return outputs

    def construct(self, inputs, text_mask):
        ''' construct '''
        embedded_inputs = self.transpose(self.embedding(inputs), (0, 2, 1))
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inferencev2(
            encoder_outputs, text_mask)

        mel_outputs_postnet = self.postnet.inference(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        return outputs


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
clip_grad = ops.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in [0, 1]:
        return grad
    dt = ops.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, ops.cast(ops.tuple_to_array(
            (-clip_value,)), dt), ops.cast(ops.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, ops.cast(ops.tuple_to_array((clip_value,)), dt))
    return new_grad


grad_scale = ops.MultitypeFuncGraph("grad_scale")
reciprocal = ops.Reciprocal()


@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    ''' scale grad '''
    return grad * ops.cast(reciprocal(scale), ops.dtype(grad))


_grad_overflow = ops.MultitypeFuncGraph("_grad_overflow")
grad_overflow = ops.FloatStatus()


@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    ''' grad overflow '''
    return grad_overflow(grad)


compute_norm = ops.MultitypeFuncGraph("compute_norm")


@compute_norm.register("Tensor")
def _compute_norm(grad):
    norm = nn.Norm()
    norm = norm(ops.cast(grad, mindspore.float32))
    ret = ops.expand_dims(ops.cast(norm, mindspore.float32), 0)
    return ret


grad_div = ops.MultitypeFuncGraph("grad_div")


@grad_div.register("Tensor", "Tensor")
def _grad_div(val, grad):
    div = ops.RealDiv()
    mul = ops.Mul()
    scale = div(1.0, val)
    ret = mul(grad, scale)
    return ret


class TrainStepWrap(nn.Cell):
    """
    TrainStepWrap definition
    """

    def __init__(self, network, optimizer, scale_update_cell):  # 16384.0
        super(TrainStepWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.add_flags(has_effect=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer

        self.hyper_map = ops.HyperMap()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)

        self.sens = 1.0
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.get_shape = ops.Shape()
        self.cast = ops.Cast()
        self.concat = ops.Concat()
        self.less_equal = ops.LessEqual()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.greater = ops.Greater()
        self.select = ops.Select()
        self.alloc_status = ops.NPUAllocFloatStatus()
        self.get_status = ops.NPUGetFloatStatus()
        self.clear_before_grad = ops.NPUClearFloatStatus()
        self.is_distributed = False
        self.norm = nn.Norm(keep_dims=True)
        self.base = Tensor(1, mindspore.float32)

        self.all_reduce = ops.AllReduce()

        self.loss_scaling_manager = scale_update_cell
        self.loss_scale = Parameter(
            Tensor(
                scale_update_cell.get_loss_scale(),
                dtype=mindspore.float32))

        self.reducer_flag = False
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [
                ParallelMode.DATA_PARALLEL,
                ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
            self.is_distributed = True
        self.grad_reducer = ops.identity
        self.degree = 1
        if self.reducer_flag:
            self.degree = get_group_size()
            mean = _get_gradients_mean()
            self.grad_reducer = DistributedGradReducer(
                self.weights, mean, self.degree)

    def construct(
        self,
        text_padded,
        input_length,
        mel_padded,
        gate_padded,
        text_mask,
        mel_mask,
        rnn_mask,
    ):
        ''' construct '''
        weights = self.weights
        loss = self.network(
            text_padded,
            input_length,
            mel_padded,
            gate_padded,
            text_mask,
            mel_mask,
            rnn_mask
        )

        scale_sense = self.loss_scale

        init = self.alloc_status()
        init = ops.depend(init, loss)

        clear_status = self.clear_before_grad(init)
        scale_sense = ops.depend(scale_sense, clear_status)
        grads = self.grad(
            self.network,
            weights)(
                text_padded,
                input_length,
                mel_padded,
                gate_padded,
                text_mask,
                mel_mask,
                rnn_mask,
                self.cast(
                    scale_sense,
                    mindspore.float32))
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(
            ops.partial(
                grad_scale,
                self.degree *
                scale_sense),
            grads)
        grads = self.hyper_map(
            ops.partial(
                clip_grad,
                GRADIENT_CLIP_TYPE,
                GRADIENT_CLIP_VALUE),
            grads)

        init = ops.depend(init, grads)
        get_status = self.get_status(init)
        init = ops.depend(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))

        if self.is_distributed:
            flag_reduce = self.all_reduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)

        overflow = self.loss_scaling_manager(self.loss_scale, cond)
        overflow = 0

        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)

        ret = (loss, scale_sense)
        return ops.depend(ret, succ)
