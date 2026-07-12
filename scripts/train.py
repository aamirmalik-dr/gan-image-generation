"""Train a DCGAN and produce the hero training GIF plus supporting artifacts.

The generator is snapshotted on a fixed batch of noise after every epoch. Those
snapshots are assembled into an animated GIF that shows the samples improve, and
the final generator is saved as a small checkpoint for instant offline sampling.

Usage:
    python scripts/train.py --dataset mnist --epochs 15 --subset 6000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dcgan.data import image_loaders
from dcgan.models import Generator
from dcgan.train import GANTrainer, sample_grid, save_generator, set_seed
from dcgan.visualize import save_grid_png, save_training_gif


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--subset", type=int, default=6000)
    parser.add_argument(
        "--ngf", type=int, default=32, help="Generator width (small keeps .pt tiny)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results", help="Directory for figures and metrics.")
    parser.add_argument("--assets", default="assets", help="Directory for the hero GIF and losses.")
    parser.add_argument("--examples", default="examples", help="Directory for example grids.")
    parser.add_argument("--model-out", default="models/generator.pt", help="Pretrained .pt path.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Per-epoch checkpoints.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    assets_dir = Path(args.assets)
    examples_dir = Path(args.examples)
    for d in (out_dir, assets_dir, examples_dir):
        d.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    loader = image_loaders(args.dataset, subset=args.subset, seed=args.seed)
    trainer = GANTrainer(generator=Generator(ngf=args.ngf))
    fixed_noise = trainer.generator.sample_noise(64)
    trainer.fit(
        loader,
        epochs=args.epochs,
        sample_every=1,  # snapshot every epoch for a smooth GIF
        fixed_noise=fixed_noise,
        verbose=True,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Hero training GIF: the fixed-noise grid evolving across epochs.
    gif_path = save_training_gif(trainer.samples, assets_dir / "training.gif")

    # Final sample grid and two example grids from different noise seeds.
    final_grid = sample_grid(trainer.generator, fixed_noise, nrow=8)
    save_grid_png(final_grid, out_dir / "samples.png")
    for i, seed in enumerate((1, 2), start=1):
        set_seed(seed)
        noise = trainer.generator.sample_noise(64)
        save_grid_png(sample_grid(trainer.generator, noise, nrow=8), examples_dir / f"grid_{i}.png")

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
    plt.savefig(assets_dir / "losses.png", dpi=120)
    plt.close()

    # Pretrained generator for instant offline sampling.
    model_path = save_generator(trainer.generator, args.model_out, epoch=args.epochs)

    # Metrics: full loss history and final values (GAN losses, not accuracy).
    metrics = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "subset": args.subset,
        "ngf": args.ngf,
        "seed": args.seed,
        "final_loss_d": round(trainer.history["loss_d"][-1], 4),
        "final_loss_g": round(trainer.history["loss_g"][-1], 4),
        "min_loss_d": round(min(trainer.history["loss_d"]), 4),
        "max_loss_d": round(max(trainer.history["loss_d"]), 4),
        "min_loss_g": round(min(trainer.history["loss_g"]), 4),
        "max_loss_g": round(max(trainer.history["loss_g"]), 4),
        "loss_d": [round(x, 4) for x in trainer.history["loss_d"]],
        "loss_g": [round(x, 4) for x in trainer.history["loss_g"]],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nWrote GIF to {gif_path}")
    print(f"Wrote generator to {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"Wrote figures and metrics to {out_dir}/ and {assets_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
