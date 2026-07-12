import torch

from dcgan.models import Discriminator, Generator, weights_init
from dcgan.train import GANTrainer, sample_grid, set_seed


def test_generator_output_shape():
    g = Generator(nz=100)
    z = g.sample_noise(4)
    out = g(z)
    assert out.shape == (4, 1, 28, 28)
    assert out.min() >= -1.0 and out.max() <= 1.0  # Tanh range


def test_discriminator_output_shape():
    d = Discriminator()
    x = torch.randn(4, 1, 28, 28)
    assert d(x).shape == (4,)


def test_sample_noise_shape():
    g = Generator(nz=64)
    assert g.sample_noise(7).shape == (7, 64, 1, 1)


def test_weights_init_runs():
    g = Generator()
    g.apply(weights_init)  # should not raise


def test_sample_grid_range_and_shape():
    g = Generator(nz=100)
    grid = sample_grid(g, g.sample_noise(16), nrow=4)
    assert grid.ndim == 2
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_trainer_one_epoch_on_random_data():
    set_seed(0)
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.rand(64, 1, 28, 28) * 2 - 1  # [-1, 1]
    y = torch.zeros(64, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, drop_last=True)
    trainer = GANTrainer()
    trainer.fit(loader, epochs=1, verbose=False)
    assert len(trainer.history["loss_g"]) == 1
    assert len(trainer.history["loss_d"]) == 1
