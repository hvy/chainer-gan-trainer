from chainer import training, cuda
from plot import savefig


@training.make_extension(trigger=(1, 'epoch'))
def sample_ims(trainer):
    x = trainer.updater.forward(test=True)
    x = x.data
    xp = cuda.get_array_module(x)
    x = xp.squeeze(x, axis=1)
    if xp == cuda.cupy:
        x = cuda.to_cpu(x)

    filename = 'result/{}.png'.format(trainer.updater.iteration)
    savefig(x, filename)
