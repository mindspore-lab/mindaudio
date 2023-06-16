"""cepstral mean and variance normalization definition."""

import mindspore


class GlobalCMVN(mindspore.nn.Cell):
    """cmvn definition.

    Args:
        mean (mindspore.Tensor): mean stats
        istd (mindspore.Tensor): inverse std, std which is 1.0 / std
        norm_var (bool): Whether variance normalization is performed, default: True
    """

    def __init__(
        self, mean: mindspore.Tensor, istd: mindspore.Tensor, norm_var: bool = True
    ):
        """Construct an CMVN object."""
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        self.mean = mean
        self.istd = istd

    def construct(self, x: mindspore.Tensor):
        """the calculation process for cmvn.

        Args:
            x (mindspore.Tensor): (batch, max_len, feat_dim)
        Returns:
            (mindspore.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x
