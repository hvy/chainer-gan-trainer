import os
from chainer import training, cuda
from chainer.training import extension
import chainer.training.trigger as trigger_module
import plot


class GeneratorSample(extension.Extension):
    def __init__(self, trigger, dirname='sample', sample_format='png'):
        self._trigger = trigger_module.get_trigger(trigger)
        self._dirname = dirname
        self._sample_format = sample_format

    def __call__(self, trainer):
        if self._trigger(trainer):
            dirname = os.path.join(trainer.out, self._dirname)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

            x = self.sample(trainer)

            filename = '{}.{}'.format(trainer.updater.epoch,
                                      self._sample_format)
            filename = os.path.join(dirname, filename)
            plot.save_ims(filename, x)

    def sample(self, trainer):
        x = trainer.updater.forward(test=True)
        x = x.data
        if cuda.get_array_module(x) == cuda.cupy:
            x = cuda.to_cpu(x)
        return x


@training.make_extension(trigger=(1, 'epoch'))
def sample_ims(trainer):
    x = trainer.updater.forward(test=True)
    x = x.data
    if cuda.get_array_module(x) == cuda.cupy:
        x = cuda.to_cpu(x)
    filename = 'result/sample/{}.png'.format(trainer.updater.epoch)
    plot.save_ims(filename, x)
