import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np


class AdditiveAngularMargin(nn.Cell):
    """
    An implementation of Additive Angular Margin (AAM) proposed

    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super(AdditiveAngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.sqrt = ms.ops.Sqrt()
        self.pow = ms.ops.Pow()

    def construct(self, outputs, targets):
        """
        Compute AAM between two tensors
        """
        cosine = outputs
        sine = self.sqrt(1.0 - self.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = np.where(cosine > 0, phi, cosine)
        else:
            phi = np.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs
