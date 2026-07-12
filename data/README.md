# Data

This directory is gitignored. No datasets are committed.

The GAN trains on MNIST or Fashion-MNIST, downloaded through torchvision on
first use:

```bash
python scripts/download_data.py --root data --dataset mnist
```

The unit tests use random tensors and need no download.
