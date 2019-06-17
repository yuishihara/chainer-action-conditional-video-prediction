import numpy as np
import chainer.functions as F
from chainer import Chain


class AutoEncoder(Chain):
    def __init__(self):
        super(AutoEncoder, self).__init__()

    def encode(self, x):
        raise NotImplementedError("encode() not implemented")

    def decode(self, h):
        raise NotImplementedError("decode() not implemented")

    def is_convolution(self):
        return False

    def loss(self, x, y):
        return F.mean_squared_error(self(x), y)

    def __call__(self, x):
        return self.decode(self.encode(x))
