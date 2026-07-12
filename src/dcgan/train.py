"""GAN training loop, sample-grid helper, and seeding."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dcgan.models import Discriminator, Generator, weights_init


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def sample_grid(generator: Generator, noise: torch.Tensor, nrow: int = 8) -> np.ndarray:
    """Generate images from ``noise`` and tile them into a single [0, 1] image.

    Args:
        generator: The generator (set to eval by this function).
        noise: Latent vectors of shape ``(n, nz, 1, 1)``.
        nrow: Number of images per row in the grid.

    Returns:
        A 2D numpy array in [0, 1] holding the tiled grid.
    """
    was_training = generator.training
    generator.eval()
    imgs = generator(noise).cpu()  # (n, 1, 28, 28) in [-1, 1]
    generator.train(was_training)
    imgs = (imgs + 1.0) / 2.0  # to [0, 1]
    n = imgs.shape[0]
    ncol = (n + nrow - 1) // nrow
    grid = np.ones((ncol * 28 + (ncol - 1), nrow * 28 + (nrow - 1)), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, nrow)
        y, x = r * 29, c * 29
        grid[y : y + 28, x : x + 28] = imgs[i, 0].numpy()
    return grid


@dataclass
class GANTrainer:
    """Trains a DCGAN with the non-saturating loss and Adam (per DCGAN)."""

    nz: int = 100
    lr: float = 2e-4
    beta1: float = 0.5
    device: str = "cpu"
    generator: Generator = field(default=None)  # type: ignore[assignment]
    discriminator: Discriminator = field(default=None)  # type: ignore[assignment]
    history: dict[str, list[float]] = field(default_factory=lambda: {"loss_g": [], "loss_d": []})

    def __post_init__(self) -> None:
        if self.generator is None:
            self.generator = Generator(nz=self.nz)
        if self.discriminator is None:
            self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def fit(
        self,
        loader: DataLoader,
        epochs: int = 20,
        sample_every: int = 0,
        fixed_noise: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> GANTrainer:
        """Train for ``epochs``.

        Args:
            loader: DataLoader of real images in [-1, 1].
            epochs: Number of epochs.
            sample_every: If > 0, store a sample grid every this many epochs in
                ``self.samples`` (requires ``fixed_noise``).
            fixed_noise: Latent vectors used for progress snapshots.
            verbose: Print per-epoch losses.
        """
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        loss_fn = nn.BCEWithLogitsLoss()
        self.samples: list[tuple[int, np.ndarray]] = []

        for epoch in range(epochs):
            eg, ed, n = 0.0, 0.0, 0
            for real, _ in loader:
                real = real.to(self.device)
                bs = real.shape[0]
                ones = torch.ones(bs, device=self.device)
                zeros = torch.zeros(bs, device=self.device)

                # Train discriminator on real and fake.
                opt_d.zero_grad()
                out_real = self.discriminator(real)
                loss_real = loss_fn(out_real, ones)
                noise = self.generator.sample_noise(bs, self.device)
                fake = self.generator(noise)
                out_fake = self.discriminator(fake.detach())
                loss_fake = loss_fn(out_fake, zeros)
                loss_d = loss_real + loss_fake
                loss_d.backward()
                opt_d.step()

                # Train generator to fool the discriminator (non-saturating).
                opt_g.zero_grad()
                out = self.discriminator(fake)
                loss_g = loss_fn(out, ones)
                loss_g.backward()
                opt_g.step()

                eg += loss_g.item() * bs
                ed += loss_d.item() * bs
                n += bs

            self.history["loss_g"].append(eg / max(n, 1))
            self.history["loss_d"].append(ed / max(n, 1))
            if verbose:
                print(
                    f"epoch {epoch + 1:3d}  loss_d={ed / max(n, 1):.4f}  "
                    f"loss_g={eg / max(n, 1):.4f}"
                )
            if sample_every and fixed_noise is not None and (epoch + 1) % sample_every == 0:
                self.samples.append((epoch + 1, sample_grid(self.generator, fixed_noise)))
        return self
