import numpy as np
import chainer
from chainer import training
from chainer import functions as F


class GenerativeAdversarialUpdater(training.StandardUpdater):
    def __init__(self, iterators, optimizers, models, converter=None, device=-1):
        super().__init__(iterators, optimizers, device=device)

        for name, optimizer in optimizers.items():
            optimizer.setup(models[name])

        if device >= 0:
            chainer.cuda.get_device(device).use()
            [model.to_gpu() for model in models.values()]

        self._models = models

        self.xp = chainer.cuda.cupy if device >= 0 else np

        self.loss_generator = 0
        self.loss_discriminator = 0

    @property
    def generator(self):
        return self._models['G']

    @property
    def discriminator(self):
        return self._models['D']

    @property
    def optimizer_generator(self):
        return self._optimizers['G']

    @property
    def optimizer_discriminator(self):
        return self._optimizers['D']


    def update_core(self):
        if self.is_new_epoch:
            print('New epoch!')
            self.loss_generator = 0
            self.loss_discriminator = 0
        else:
            print('hm, ok!')

        batch = self._iterators['noise'].next()
        in_arrays = self.converter(batch, self.device)

        print('noise shape')
        print(in_arrays.shape)

        # TODO: wrap in Variable if necessary
        x_fake = self.generator(in_arrays)
        y_fake = self.discriminator(x_fake)

        print('generated shape')
        print(x_fake.shape)
        print('y_fake shape')
        print(y_fake.shape)

        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        print('real data shape')
        print(in_arrays.shape)
        y_real = self.discriminator(in_arrays)

        print('y_real shape')
        print(y_real.shape)

        generator_loss = F.softmax_cross_entropy(y_fake, self.xp.ones(y_fake.shape[0], dtype=self.xp.int32))
        discriminator_loss = F.softmax_cross_entropy(y_fake, self.xp.zeros(y_fake.shape[0], dtype=self.xp.int32))
        discriminator_loss += F.softmax_cross_entropy(y_real, self.xp.ones(y_real.shape[0], dtype=self.xp.int32))
        discriminator_loss /= 2

        for optimizer in self._optimizers.values():
            optimizer.target.cleargrads()

        discriminator_loss.backward()
        print(self._optimizers['D'])
        print(self._optimizers['D'].beta1)
        self._optimizers['D'].update()

        generator_loss.backward()
        self._optimizers['G'].update()
