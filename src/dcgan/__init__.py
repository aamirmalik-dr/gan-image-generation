"""A deep convolutional GAN (DCGAN) for 28x28 grayscale images.

An original generator and discriminator, a training loop with the standard
non-saturating GAN loss, and utilities to sample and lay out image grids. The
demo trains on MNIST or Fashion-MNIST.
"""

from dcgan.data import image_loaders
from dcgan.models import Discriminator, Generator, weights_init
from dcgan.train import GANTrainer, sample_grid, set_seed

__all__ = [
    "Generator",
    "Discriminator",
    "weights_init",
    "image_loaders",
    "GANTrainer",
    "sample_grid",
    "set_seed",
]

__version__ = "0.1.0"
