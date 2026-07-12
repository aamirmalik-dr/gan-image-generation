"""Generator and discriminator for 28x28 grayscale images.

The generator maps a latent vector to a 1x28x28 image through transposed
convolutions; the discriminator maps an image to a single real/fake logit. Both
follow the DCGAN design guidelines (strided convolutions, batch norm, ReLU in
the generator and LeakyReLU in the discriminator) but are written from scratch
and sized for MNIST rather than 64x64 images.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Map a latent vector ``z`` of shape ``(N, nz, 1, 1)`` to a 1x28x28 image."""

    def __init__(self, nz: int = 100, ngf: int = 64) -> None:
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            # nz x 1 x 1 -> (ngf*4) x 7 x 7
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # -> (ngf*2) x 14 x 14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> 1 x 28 x 28
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def sample_noise(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Return ``n`` latent vectors of shape ``(n, nz, 1, 1)``."""
        return torch.randn(n, self.nz, 1, 1, device=device)


class Discriminator(nn.Module):
    """Map a 1x28x28 image to a single real/fake logit."""

    def __init__(self, ndf: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # 1 x 28 x 28 -> ndf x 14 x 14
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*2) x 7 x 7
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> 1 x 1 x 1
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def weights_init(module: nn.Module) -> None:
    """Initialize conv and batch-norm weights per the DCGAN paper."""
    name = module.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
