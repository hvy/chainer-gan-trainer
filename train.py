import argparse
from chainer import datasets, training, iterators, optimizers
from chainer.training import updater, extensions
from iterators import RandomNoiseIterator, UniformNoiseGenerator
from models import Generator, Discriminator
from updater import GenerativeAdversarialUpdater
from extensions import GeneratorSample


iterators.RandomNoiseIterator = RandomNoiseIterator
updater.GenerativeAdversarialUpdater = GenerativeAdversarialUpdater
extensions.GeneratorSample = GeneratorSample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    nz = args.nz
    batch_size = args.batch_size
    epochs = args.epochs
    gpu = args.gpu

    train, _ = datasets.get_mnist(withlabel=False, ndim=3)
    train_iter = iterators.SerialIterator(train, batch_size)
    z_iter = iterators.RandomNoiseIterator(UniformNoiseGenerator(-1, 1, nz), batch_size)

    optimizer_generator = optimizers.Adam(alpha=1e-3, beta1=0.5)
    optimizer_discriminator = optimizers.Adam(alpha=2e-4, beta1=0.5)

    optimizer_generator.setup(Generator(nz))
    optimizer_discriminator.setup(Discriminator(train.shape[2:]))

    updater = updater.GenerativeAdversarialUpdater(
        iterator=train_iter,
        noise_iterator=z_iter,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        device=gpu)

    trainer = training.Trainer(updater, stop_trigger=(epochs, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'gen/loss', 'dis/loss']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.GeneratorSample(trigger=(1, 'epoch')))
    trainer.run()
