# Chainer GAN Trainer Demo

Implementation of GAN with Chainer's *trainer* using the following custom classes.

- `chainer.iterators`
- `chainer.training.updater`
- `chainer.training.extensions`

## Custom Trainer Components

### chainer.iterators.RandomNoiseIterator

Iterator that keeps producing random arrays of Gaussian or uniform distribution. It has no notion of epochs.

### chainer.training.updater.GenerativeAdversarialUpdater

Updater responsible for the GAN training algorithm including forward pass, backward pass and parameter updates.

### chainer.training.extensions.GeneratorSample

Extension that automatically takes random sample images and saved them to disk at any given interval.

## Run

### Train

```bash
python train.py --gpu 0
```
