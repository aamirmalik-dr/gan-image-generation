"""A deep convolutional GAN (DCGAN) for 28x28 grayscale images.

An original generator and discriminator, a training loop with the standard
non-saturating GAN loss, and utilities to sample and lay out image grids. The
demo trains on MNIST or Fashion-MNIST.
"""

from dcgan.data import image_loaders
from dcgan.models import Discriminator, Generator, weights_init
from dcgan.train import (
    GANTrainer,
    load_generator,
    sample_grid,
    save_generator,
    set_seed,
)
from dcgan.visualize import save_grid_png, save_training_gif

__all__ = [
    "Generator",
    "Discriminator",
    "weights_init",
    "image_loaders",
    "GANTrainer",
    "sample_grid",
    "set_seed",
    "save_generator",
    "load_generator",
    "save_grid_png",
    "save_training_gif",
]

__version__ = "0.1.0"
