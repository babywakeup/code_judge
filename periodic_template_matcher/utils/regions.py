"""Original-coordinate valid-region policy shared by all matcher stages."""
from dataclasses import dataclass
from math import isfinite

import numpy as np
import cv2


@dataclass(frozen=True)
class ValidRegion:
    """Valid top rows [0, valid_height), intersected with an optional image mask.

    Height is in original-image pixels and overrides the legacy bottom fraction.
    The full-size mask preserves image coordinates, labels and output dimensions.
    """

    valid_height: int | None = None
    bottom_ignore_fraction: float = 0.

    @classmethod
    def from_config(cls, config, valid_height=None):
        height = config.valid_height if valid_height is None else valid_height
        return cls(height, config.bottom_ignore_fraction)

    def resolved_height(self, image):
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
            raise ValueError('ValidRegion requires a nonempty image array')
        height = image.shape[0]
        if (not isinstance(self.bottom_ignore_fraction, (int, float))
                or not isfinite(self.bottom_ignore_fraction)
                or not 0 <= self.bottom_ignore_fraction < .5):
            raise ValueError('bottom_ignore_fraction must be finite in [0, .5)')
        if self.valid_height is not None:
            if type(self.valid_height) is not int or not 1 <= self.valid_height <= height:
                raise ValueError(f'valid_height must be an integer in 1..{height}')
            return self.valid_height
        return int(height*(1-self.bottom_ignore_fraction))

    def build_mask(self, image, mask=None):
        """Return a new HxW uint8 image ROI, never annotation knowledge."""
        end = self.resolved_height(image)
        if mask is None:
            out = np.ones(image.shape[:2], np.uint8)
        else:
            if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]:
                raise ValueError('Valid-mask size must match image height and width')
            out = (mask > 0).astype(np.uint8)
        out[end:] = 0
        if out.sum() < 1024:
            raise ValueError('Too few valid image pixels')
        return out

    def crop(self, image):
        """Return a top-row image copy; optional convenience for external callers."""
        return image[:self.resolved_height(image)].copy()

    @staticmethod
    def resize_for_matching(gray, mask, size):
        """Downsample using only valid pixels, even across fractional ROI borders.

        Returns (uint8 gray, uint8 mask). Multiplication happens before resize,
        preventing excluded text from entering a partly valid output pixel.
        """
        weights = (mask > 0).astype(np.float32)
        resized_mask = cv2.resize(weights, size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        if weights.all():
            return cv2.resize(gray, size, interpolation=cv2.INTER_AREA), resized_mask
        numerator = cv2.resize(gray.astype(np.float32)*weights, size, interpolation=cv2.INTER_AREA)
        denominator = cv2.resize(weights, size, interpolation=cv2.INTER_AREA)
        resized = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-6)
        resized_mask &= (denominator > 1e-6).astype(np.uint8)
        return np.clip(np.rint(resized), 0, 255).astype(np.uint8), resized_mask

    def describe(self, image, mask):
        """JSON-safe record of the resolved image region, including mask holes."""
        return dict(image_height=int(image.shape[0]), image_width=int(image.shape[1]),
                    valid_height=self.resolved_height(image), valid_pixels=int(mask.sum()),
                    height_source='explicit' if self.valid_height is not None else 'bottom_fraction')
