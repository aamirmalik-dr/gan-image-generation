"""GAN training loop, sample-grid helper, checkpointing, and seeding."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dcgan.models import Discriminator, Generator, weights_init

logger = logging.getLogger("dcgan")


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_generator(generator: Generator, path: str | Path, epoch: int | None = None) -> Path:
    """Save a generator to ``path`` with the config needed to rebuild it.

    The checkpoint is a plain dict of the state dict plus ``nz``, ``ngf``, and
    the epoch, so :func:`load_generator` can reconstruct the architecture
    without any external metadata.

    Args:
        generator: The generator to serialize.
        path: Destination ``.pt`` file. Parent directories are created.
        epoch: Optional epoch tag stored in the checkpoint.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": generator.state_dict(),
            "nz": generator.nz,
            "ngf": generator.ngf,
            "epoch": epoch,
        },
        path,
    )
    return path


def load_generator(path: str | Path, device: str = "cpu") -> Generator:
    """Rebuild a generator from a checkpoint written by :func:`save_generator`.

    Args:
        path: A ``.pt`` checkpoint file.
        device: Device to place the generator on.

    Returns:
        A generator in eval mode with weights loaded.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    generator = Generator(nz=int(ckpt["nz"]), ngf=int(ckpt["ngf"]))
    generator.load_state_dict(ckpt["state_dict"])
    generator.to(device)
    generator.eval()
    return generator


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
        checkpoint_dir: str | Path | None = None,
    ) -> GANTrainer:
        """Train for ``epochs``.

        Args:
            loader: DataLoader of real images in [-1, 1].
            epochs: Number of epochs.
            sample_every: If > 0, store a sample grid every this many epochs in
                ``self.samples`` (requires ``fixed_noise``).
            fixed_noise: Latent vectors used for progress snapshots.
            verbose: Print per-epoch losses.
            checkpoint_dir: If set, save the generator to
                ``generator_epoch_XX.pt`` in this directory after each epoch and
                log progress there.
        """
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
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

            mean_g = eg / max(n, 1)
            mean_d = ed / max(n, 1)
            self.history["loss_g"].append(mean_g)
            self.history["loss_d"].append(mean_d)
            msg = f"epoch {epoch + 1:3d}  loss_d={mean_d:.4f}  loss_g={mean_g:.4f}"
            if verbose:
                print(msg)
            logger.info(msg)
            if sample_every and fixed_noise is not None and (epoch + 1) % sample_every == 0:
                self.samples.append((epoch + 1, sample_grid(self.generator, fixed_noise)))
            if ckpt_dir is not None:
                save_generator(
                    self.generator, ckpt_dir / f"generator_epoch_{epoch + 1:02d}.pt", epoch + 1
                )
        return self
