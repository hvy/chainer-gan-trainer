import numpy as np
import chainer
from chainer import training, reporter
from chainer import functions as F
from chainer import Variable


class GenerativeAdversarialUpdater(training.StandardUpdater):
    def __init__(self, *, iterator, noise_iterator, optimizer_generator,
                 optimizer_discriminator, converter=None, device=-1):

        iterators = {'main': iterator, 'z': noise_iterator}
        optimizers = {'gen': optimizer_generator,
                      'dis': optimizer_discriminator}

        super().__init__(iterators, optimizers, device=device)

        if device >= 0:
            chainer.cuda.get_device(device).use()
            [optimizer.target.to_gpu() for optimizer in optimizers.values()]

        self.xp = chainer.cuda.cupy if device >= 0 else np

    @property
    def generator(self):
        return self._optimizers['gen'].target

    @property
    def discriminator(self):
        return self._optimizers['dis'].target

    def forward(self):
        z = self._iterators['z'].next()
        z = self.converter(z, self.device)

        x_fake = self.generator(Variable(z))
        y_fake = self.discriminator(x_fake)

        x_real = self._iterators['main'].next()
        x_real = self.converter(x_real, self.device)

        y_real = self.discriminator(Variable(x_real))

        return y_fake, y_real

    def backward(self, y):
        y_fake, y_real = y

        generator_loss = F.softmax_cross_entropy(
            y_fake,
            Variable(self.xp.ones(y_fake.shape[0], dtype=self.xp.int32)))
        discriminator_loss = F.softmax_cross_entropy(
            y_fake,
            Variable(self.xp.zeros(y_fake.shape[0], dtype=self.xp.int32)))
        discriminator_loss += F.softmax_cross_entropy(
            y_real,
            Variable(self.xp.ones(y_real.shape[0], dtype=self.xp.int32)))
        discriminator_loss /= 2

        return {'gen': generator_loss, 'dis': discriminator_loss}

    def update_params(self, losses, report=True):
        for optimizer in self._optimizers.values():
            optimizer.target.cleargrads()

        for name, loss in losses.items():
            if report:
                reporter.report({'{}/loss'.format(name): loss})

            loss.backward()
            self._optimizers[name].update()

    def update_core(self):
        if self.is_new_epoch:
            pass

        losses = self.backward(self.forward())
        self.update_params(losses, report=True)
