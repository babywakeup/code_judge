"""Unicode-safe image IO and photograph validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import Array, Image, PathInput


def read_image(path: PathInput, flags: int = cv2.IMREAD_COLOR) -> Array:
    """Read an image, including Windows paths containing non-ASCII characters."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def write_image(path: PathInput, image: Array) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    buf.tofile(str(path))


def validate_image(image: Image) -> None:
    """Check photograph shape/dtype without allocating a gray conversion."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError('Empty image')
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError('Expected gray, BGR or BGRA image')
    if image.ndim not in (2, 3) or min(image.shape[:2]) < 32:
        raise ValueError('Image must be at least 32 x 32')
    if image.dtype != np.uint8:
        raise ValueError('Input photographs must be uint8; normalize other depths explicitly')


def to_gray(image: Image) -> Image:
    """Validate a uint8 photograph and convert gray/BGR/BGRA to gray."""
    validate_image(image)
    if image.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)
    return image
