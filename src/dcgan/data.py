"""Image data loading for the GAN.

Images are scaled to [-1, 1] to match the generator's Tanh output.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset

DATASETS = ("mnist", "fashion")


def image_loaders(
    dataset: str = "mnist",
    root: str = "data",
    batch_size: int = 128,
    subset: int | None = 12000,
    seed: int = 0,
) -> DataLoader:
    """Return a single training DataLoader of images scaled to [-1, 1].

    Args:
        dataset: ``"mnist"`` or ``"fashion"``.
        root: Directory for the torchvision download (gitignored).
        batch_size: Batch size.
        subset: If set, use this many training images (keeps the demo fast).
        seed: Seed for the subsample.

    Returns:
        A training ``DataLoader``.

    Raises:
        ValueError: If ``dataset`` is not recognized.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
    train = cls(root=root, train=True, download=True, transform=tf)

    if subset is not None:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train), size=min(subset, len(train)), replace=False)
        train = Subset(train, idx.tolist())

    return DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
