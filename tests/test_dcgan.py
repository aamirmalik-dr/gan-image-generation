import numpy as np
import torch

from dcgan.models import Discriminator, Generator, weights_init
from dcgan.train import (
    GANTrainer,
    load_generator,
    sample_grid,
    save_generator,
    set_seed,
)
from dcgan.visualize import save_grid_png, save_training_gif


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


def test_save_and_load_generator_roundtrip(tmp_path):
    g = Generator(nz=64, ngf=32)
    path = save_generator(g, tmp_path / "g.pt", epoch=3)
    assert path.exists()
    loaded = load_generator(path)
    assert loaded.nz == 64 and loaded.ngf == 32
    z = loaded.sample_noise(2)
    a = g.eval()(z)
    b = loaded(z)
    assert torch.allclose(a, b, atol=1e-5)  # weights survived the round trip


def test_trainer_writes_checkpoints(tmp_path):
    set_seed(0)
    from torch.utils.data import DataLoader, TensorDataset

    x = torch.rand(64, 1, 28, 28) * 2 - 1
    y = torch.zeros(64, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, drop_last=True)
    trainer = GANTrainer(generator=Generator(ngf=16))
    trainer.fit(loader, epochs=2, verbose=False, checkpoint_dir=tmp_path)
    saved = sorted(tmp_path.glob("generator_epoch_*.pt"))
    assert len(saved) == 2
    assert load_generator(saved[-1]).ngf == 16


def test_save_grid_png(tmp_path):
    g = Generator(ngf=16)
    grid = sample_grid(g, g.sample_noise(4), nrow=2)
    path = save_grid_png(grid, tmp_path / "grid.png", scale=2)
    assert path.exists() and path.stat().st_size > 0


def test_save_training_gif(tmp_path):
    frames = [(e, np.random.rand(29, 29).astype(np.float32)) for e in (1, 2, 3)]
    path = save_training_gif(frames, tmp_path / "train.gif", scale=2, duration_ms=100)
    assert path.exists() and path.stat().st_size > 0
