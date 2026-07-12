# gan-image-generation

A deep convolutional GAN (DCGAN) in PyTorch that generates handwritten-digit images from random noise. An original generator and discriminator, the standard non-saturating GAN training loop, and per-epoch sample grids that show the generator improve over training.

## What it does

- Implements a generator (transposed convolutions, batch norm, ReLU, Tanh output) and a discriminator (strided convolutions, batch norm, LeakyReLU) from scratch, sized for 28x28 grayscale images.
- Trains both networks together with the non-saturating GAN loss and the Adam settings from the DCGAN paper, on MNIST or Fashion-MNIST scaled to [-1, 1].
- Snapshots a fixed batch of noise at several epochs to visualize learning progress, and writes a final sample grid, a progression grid, and generator/discriminator loss curves.
- Ships unit tests that run on random tensors with no download, checking output shapes, the Tanh range, and that a training step runs.

## What it does not do

- No FID or Inception Score. Sample quality is shown qualitatively through the generated grids, not scored with a pretrained metric network.
- No conditional generation, no progressive growing, no large or high-resolution images. The demo targets 28x28 digits so it trains on a CPU in minutes.
- No stability tricks beyond the DCGAN recipe (no spectral norm, gradient penalty, or two-timescale update rule).

## Install

```
python -m venv .venv
.venv\Scripts\activate      # Windows, or: source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11 or newer. On Linux CI, install the CPU build of PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Run

```
python scripts/download_data.py --root data --dataset mnist   # optional prefetch
python scripts/train.py --dataset mnist --epochs 20           # train and save figures
pytest -q                                                     # tests, fully offline
```

The demo notebook is `notebooks/demo.ipynb`, executed with saved outputs.

## Results

Produced by `python scripts/train.py --dataset mnist --epochs 20 --subset 12000`: a DCGAN trained on a 12000-image MNIST subset, batch size 128, single fixed seed, on a CPU.

The two losses stay in a stable equilibrium across all 20 epochs rather than one network collapsing: the discriminator loss ranges roughly 0.43 to 0.82 and the generator loss roughly 1.52 to 2.11, with neither running to zero. That balance is the sign of healthy adversarial training, and it is visible in the saved loss curve.

By the final epoch the generator produces a grid of 64 samples in which most images are clearly recognizable handwritten digits (zeros, fives, sixes, sevens, eights, and nines are all legible), with a minority still malformed or ambiguous, which is the expected quality for a compact DCGAN trained for 20 epochs on a subset. The progression grid shows the samples move from near-noise at the first snapshot toward digit-like strokes by the end. Longer training and the full dataset sharpen the samples further. The figures are written to `results/` by the training script and are also embedded, executed, in the demo notebook.

## Package layout

```
src/dcgan/          library code (generator and discriminator, data, trainer)
scripts/            download_data.py, train.py
notebooks/          demo.ipynb with executed outputs
tests/              pytest suite, runs on random tensors offline
data/               gitignored, MNIST downloaded on demand
```

## References

- Radford, Metz, Chintala, Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, 2016 (DCGAN).
- Goodfellow et al., Generative Adversarial Nets, 2014.

## Author

Aamir Malik

- GitHub: https://github.com/aamirmalik-dr
- LinkedIn: https://linkedin.com/in/dr-aamirmalik

## License

MIT, see LICENSE.
