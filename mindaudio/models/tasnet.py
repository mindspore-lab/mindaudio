""" TasNet """
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

EPS = 1e-8


class TasNet(nn.Cell):
    """ TasNet """

    def __init__(self, L, N, hidden_size, num_layers, bidirectional=False, nspk=2):
        super(TasNet, self).__init__()
        # hyper-parameter
        self.L = L
        self.N = N
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.nspk = nspk
        # Components
        self.encoder = Encoder(L, N)
        self.separator = Separator(
            N, hidden_size, num_layers, bidirectional=bidirectional, nspk=nspk
        )
        self.decoder = Decoder(N, L)
        for p in self.get_parameters():
            if p.ndim > 1:
                mindspore.common.initializer.Uniform(p)

    def construct(self, mixture):
        """
        Args:
            mixture: [B, K, L]
        Returns:
            est_source: [B, nspk, K, L]
        """
        mixture_w, norm_coef = self.encoder(mixture.astype(mindspore.float16))
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask, norm_coef)
        return est_source


class Encoder(nn.Cell):
    """ Encoder """

    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # hyper-parameter
        self.L = L
        self.N = N
        # Components
        self.conv1d_U = nn.Conv1d(
            L,
            N,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            has_bias=True,
            weight_init="XavierUniform",
        )
        self.conv1d_V = nn.Conv1d(
            L,
            N,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            has_bias=True,
            weight_init="XavierUniform",
        )
        self.relu = ops.ReLU()
        self.sigmoid = ops.Sigmoid()
        self.expand_dims = ops.ExpandDims()
        self.Norm = nn.Norm(axis=2, keep_dims=True)

    def construct(self, mixture):
        """
        Args:
            mixture: [B, K, L]
        Returns:
            mixture_w: [B, K, N]
            norm_coef: [B, K, 1]
        """
        B, K, L = mixture.shape
        # L2 Norm along L axis
        norm_coef = self.Norm(mixture)  # B x K x 1
        norm_mixture = mixture / (norm_coef + EPS)  # B x K x L
        # 1-D gated conv
        norm_mixture = self.expand_dims(norm_mixture.view(-1, L), 2)  # B*K x L x 1
        conv = self.relu(self.conv1d_U(norm_mixture))  # B*K x N x 1
        gate = self.sigmoid(self.conv1d_V(norm_mixture))  # B*K x N x 1
        mixture_w = conv * gate  # B*K x N x 1
        mixture_w = mixture_w.view(B, K, self.N)  # B x K x N
        return mixture_w, norm_coef


class Separator(nn.Cell):
    """ Estimation of source masks """

    def __init__(self, N, hidden_size, num_layers, bidirectional=False, nspk=2):
        super(Separator, self).__init__()
        # hyper-parameter
        self.N = N
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.nspk = nspk
        # Components
        self.layer_norm = nn.LayerNorm([N])
        self.lstm = nn.LSTM(
            N, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional
        )
        # self.fc = nn.Linear(hidden_size, nspk * N)
        self.fc = nn.Dense(hidden_size, nspk * N, weight_init="XavierUniform")
        # self.fc = nn.Dense(hidden_size, nspk * N)
        self.softmax = ops.Softmax(axis=2)

    def construct(self, mixture_w):
        """
        Args:
            mixture_w: [B, K, N], padded
        Returns:
            est_mask: [B, K, nspk, N]
        """
        B, K, N = mixture_w.shape
        # print("B size = ", B)
        # print("K size = ", K)
        # print("N size = ", N)
        # layer norm
        norm_mixture_w = self.layer_norm(mixture_w.astype(mindspore.float16))
        # norm_mixture_w = nn.LayerNorm(mixture_w[-1:])
        # LSTM
        # output, _ = self.lstm(norm_mixture_w)
        output = norm_mixture_w
        # fc
        score = self.fc(output)  # B x K x nspk*N
        score = score.view(B, K, self.nspk, N)
        # softmax
        est_mask = self.softmax(score)
        return est_mask


class Decoder(nn.Cell):
    """ Decoder """

    def __init__(self, N, L):
        super(Decoder, self).__init__()
        # hyper-parameter
        self.N, self.L = N, L
        # Components
        # self.basis_signals = nn.Linear(N, L, bias=False)
        self.basis_signals = nn.Dense(N, L, weight_init="XavierUniform")
        # self.basis_signals = nn.Dense(N, L)
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()

    def construct(self, mixture_w, est_mask, norm_coef):
        """
        Args:
            mixture_w: [B, K, N]
            est_mask: [B, K, nspk, N]
            norm_coef: [B, K, 1]
        Returns:
            est_source: [B, nspk, K, L]
        """
        # D = W * M
        source_w = self.expand_dims(mixture_w, 2) * est_mask  # B x K x nspk x N
        # S = DB
        est_source = self.basis_signals(source_w)  # B x K x nspk x L
        # reverse L2 norm
        norm_coef = self.expand_dims(norm_coef, 2)  # B x K x 1 x1
        est_source = est_source * norm_coef  # B x K x nspk x L
        est_source = self.transpose(est_source, (0, 2, 1, 3))  # B x nspk x K x L
        return est_source
