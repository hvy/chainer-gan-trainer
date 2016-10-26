import numpy
from chainer.dataset.iterator import Iterator


def to_tuple(x):
    if isinstance(x, (tuple, list)):
        return x
    else:
        return x,


class UniformNoiseGenerator(object):
    def __init__(self, low, high, size):
        self.low = low
        self.high = high
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return numpy.random.uniform(self.low, self.high,
                                      (batch_size,) +  self.size)


class GaussianNoiseGenerator(object):
    def __init__(self, loc, scale, size):
        self.loc = loc
        self.scale = scale
        self.size = to_tuple(size)

    def __call__(self, batch_size):
        return numpy.random.normal(self.loc, self.scale,
                                     (batch_size,) + self.size)


class RandomNoiseIterator(Iterator):
    def __init__(self, noise_generator, batch_size):
        self.batch_size = batch_size
        self.noise_generator = noise_generator

    def __next__(self):
        batch = self.noise_generator(self.batch_size)
        batch = batch.astype(numpy.float32)
        print('RandomNoiseIterator')
        print(batch.shape)
        return batch
