"""Turn sample grids into PNG images and an animated training GIF.

These helpers take the 2D float grids produced by
:func:`dcgan.train.sample_grid` (values in ``[0, 1]``) and render them with
Pillow. They avoid matplotlib so the animation stays small and dependency
light.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_CAPTION_H = 18


def _grid_to_gray(grid: np.ndarray, scale: int) -> Image.Image:
    """Convert a ``[0, 1]`` grid to an upscaled grayscale ``Image``."""
    arr = np.clip(grid, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    return img


def save_grid_png(grid: np.ndarray, path: str | Path, scale: int = 4) -> Path:
    """Write a single sample grid to a grayscale PNG.

    Args:
        grid: A 2D array in ``[0, 1]`` from :func:`dcgan.train.sample_grid`.
        path: Destination ``.png`` file. Parent directories are created.
        scale: Nearest-neighbour upscaling factor for readability.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _grid_to_gray(grid, scale).save(path)
    return path


def _captioned_frame(grid: np.ndarray, text: str, scale: int) -> Image.Image:
    """Render one GIF frame: the upscaled grid with a caption bar below."""
    base = _grid_to_gray(grid, scale).convert("RGB")
    frame = Image.new("RGB", (base.width, base.height + _CAPTION_H), (0, 0, 0))
    frame.paste(base, (0, 0))
    draw = ImageDraw.Draw(frame)
    draw.text((4, base.height + 3), text, fill=(255, 255, 255))
    return frame


def save_training_gif(
    frames: list[tuple[int, np.ndarray]],
    path: str | Path,
    scale: int = 3,
    duration_ms: int = 500,
    hold_last_ms: int = 1500,
) -> Path:
    """Assemble per-epoch sample grids into an animated GIF.

    Args:
        frames: A list of ``(epoch, grid)`` pairs, in order. Each grid is a 2D
            array in ``[0, 1]``.
        path: Destination ``.gif`` file. Parent directories are created.
        scale: Nearest-neighbour upscaling factor for each frame.
        duration_ms: Display time per frame.
        hold_last_ms: Extra hold time on the final frame so the result is
            readable before the loop restarts.

    Returns:
        The path written to.

    Raises:
        ValueError: If ``frames`` is empty.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [_captioned_frame(g, f"epoch {ep}", scale) for ep, g in frames]
    # Content is grayscale, so a small palette keeps the file compact.
    images = [im.quantize(colors=16, method=Image.MEDIANCUT) for im in images]
    durations = [duration_ms] * len(images)
    durations[-1] = duration_ms + hold_last_ms
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return path
