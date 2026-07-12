"""Generate a grid of images from a committed generator, fully offline.

This is the project's quickstart entry point. It loads a pretrained generator
checkpoint, samples a batch of fixed noise, and writes an 8x8 sample grid to a
PNG. No dataset download and no training are involved, so it runs in a second
on a CPU with no network.

Usage:
    python scripts/generate.py
    python scripts/generate.py --weights models/generator.pt --n 64 --seed 7 \
        --out results/generated.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dcgan.train import load_generator, sample_grid, set_seed
from dcgan.visualize import save_grid_png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="models/generator.pt",
        help="Path to a committed generator checkpoint.",
    )
    parser.add_argument("--n", type=int, default=64, help="Number of images to sample.")
    parser.add_argument("--nrow", type=int, default=8, help="Images per row in the grid.")
    parser.add_argument("--seed", type=int, default=0, help="Noise seed for reproducibility.")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor for the PNG.")
    parser.add_argument("--out", default="results/generated.png", help="Output PNG path.")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        parser.error(
            f"generator checkpoint not found: {weights}. "
            "Pass --weights or train one with scripts/train.py."
        )

    set_seed(args.seed)
    generator = load_generator(weights)
    noise = generator.sample_noise(args.n)
    grid = sample_grid(generator, noise, nrow=args.nrow)
    out = save_grid_png(grid, args.out, scale=args.scale)
    print(f"Wrote {args.n} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
