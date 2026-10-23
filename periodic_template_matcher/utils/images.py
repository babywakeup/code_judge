"""Unicode-safe image IO and photograph validation."""
from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    """Read an image, including Windows paths containing non-ASCII characters."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    buf.tofile(str(path))


def to_gray(image):
    """Validate a uint8 photograph and convert gray/BGR/BGRA to gray."""
    if image is None or image.size == 0:
        raise ValueError('Empty image')
    if image.ndim == 3:
        if image.shape[2] not in (3, 4):
            raise ValueError('Expected gray, BGR or BGRA image')
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError('Image must be at least 32 x 32')
    if image.dtype != np.uint8:
        raise ValueError('Input photographs must be uint8; normalize other depths explicitly')
    return image