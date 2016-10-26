import argparse
import chainer
from chainer import datasets, training, iterators, optimizers
from chainer import training
from chainer.training import extensions
from models import Generator, Discriminator
from updaters import GenerativeAdversarialUpdater
from iterators import RandomNoiseIterator
from iterators import UniformNoiseGenerator, GaussianNoiseGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    nz = args.nz
    batch_size = args.batch_size
    epochs = args.epochs
    gpu = args.gpu

    train, _ = datasets.get_mnist(withlabel=False, ndim=3)
    train_iter = iterators.SerialIterator(train, batch_size, repeat=True, shuffle=True)

    noise_iter = RandomNoiseIterator(UniformNoiseGenerator(-1, 1, nz), batch_size)

    iters = {'main': train_iter, 'noise': noise_iter}

    generator = Generator(nz, train.shape[2:])
    discriminator = Discriminator(train.shape[2:])

    optimizer_generator = optimizers.Adam(alpha=1e-3, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=2e-4, beta1=0.5)

    models = {'G': generator, 'D': discriminator}
    opts = {'G': optimizer_generator, 'D': optimizer_discriminator}

    updater = GenerativeAdversarialUpdater(iters, opts, models, device=gpu)
    trainer = training.Trainer(updater, (epochs, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'dis/loss', 'gen/loss']))
    trainer.run()
    print('Done')
