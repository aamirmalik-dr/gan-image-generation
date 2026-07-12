"""Train a DCGAN and write a sample grid, a progression grid, and loss curves.

Usage:
    python scripts/train.py --dataset mnist --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcgan.data import image_loaders
from dcgan.train import GANTrainer, sample_grid, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--subset", type=int, default=12000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    loader = image_loaders(args.dataset, subset=args.subset)
    trainer = GANTrainer()
    fixed_noise = trainer.generator.sample_noise(64)
    snap_every = max(1, args.epochs // 4)
    trainer.fit(
        loader,
        epochs=args.epochs,
        sample_every=snap_every,
        fixed_noise=fixed_noise,
        verbose=True,
    )

    # Final 8x8 sample grid.
    grid = sample_grid(trainer.generator, fixed_noise, nrow=8)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    plt.title(f"DCGAN samples ({args.dataset}, epoch {args.epochs})")
    plt.tight_layout()
    plt.savefig(out_dir / "samples.png", dpi=120)
    plt.close()

    # Progression across training.
    if trainer.samples:
        k = len(trainer.samples)
        fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.2))
        if k == 1:
            axes = [axes]
        for ax, (ep, g) in zip(axes, trainer.samples, strict=False):
            ax.imshow(g, cmap="gray")
            ax.set_title(f"epoch {ep}")
            ax.axis("off")
        fig.suptitle("Generator progress")
        plt.tight_layout()
        plt.savefig(out_dir / "progression.png", dpi=120)
        plt.close()

    # Loss curves.
    plt.figure(figsize=(7, 4))
    ep = range(1, len(trainer.history["loss_g"]) + 1)
    plt.plot(ep, trainer.history["loss_d"], label="discriminator")
    plt.plot(ep, trainer.history["loss_g"], label="generator")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("GAN training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "losses.png", dpi=120)
    plt.close()

    print(f"\nWrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
