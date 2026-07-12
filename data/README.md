# Data

This directory is gitignored. No datasets are committed here.

## The quickstart needs no data

The quickstart, `python scripts/generate.py`, samples images from the committed
generator weights at `models/generator.pt`. Those weights are the only thing it
reads, so it runs fully offline with no dataset present. The example grids under
`examples/` and the training GIF under `assets/` were produced the same way,
from generator weights, so nothing in the repo redistributes any dataset images.

## Training needs MNIST or Fashion-MNIST

To retrain and regenerate the GIF, the training script pulls MNIST or
Fashion-MNIST through torchvision on first use:

```bash
python scripts/download_data.py --root data --dataset mnist
python scripts/train.py --dataset mnist --epochs 15 --subset 6000
```

Both datasets are standard public research benchmarks distributed by
torchvision. They are downloaded on demand into this gitignored directory and
never committed.

The unit tests use random tensors and need no download.
