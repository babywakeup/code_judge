"""Original-coordinate valid-region policy shared by all matcher stages."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import MatcherConfig
from .types import Diagnostics, Image, Mask, Size


@dataclass(frozen=True)
class ValidRegion:
    """An axis-aligned, half-open rectangle in original image pixels."""

    valid_height: int | None = None
    x_start: int = 0
    x_end: int | None = None
    y_start: int = 0

    @classmethod
    def from_config(cls, config: MatcherConfig, valid_height: int | None = None) -> ValidRegion:
        end = config.valid_y_end if config.valid_y_end is not None else config.valid_height
        height = end if valid_height is None else valid_height
        return cls(height, config.valid_x_start, config.valid_x_end, config.valid_y_start)

    def resolved_height(self, image: Image) -> int:
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or min(image.shape[:2]) < 1:
            raise ValueError('ValidRegion requires a nonempty image array')
        height = image.shape[0]
        if self.valid_height is not None:
            if type(self.valid_height) is not int or not 1 <= self.valid_height <= height:
                raise ValueError(f'valid_height must be an integer in 1..{height}')
            return self.valid_height
        return height

    def slices(self, image: Image) -> tuple[slice, slice]:
        y_end = self.resolved_height(image)
        width = image.shape[1]
        x_end = width if self.x_end is None else self.x_end
        if (type(self.x_start) is not int or type(x_end) is not int
                or not 0 <= self.x_start < x_end <= width):
            raise ValueError(f'Expected 0 <= valid_x_start < valid_x_end <= {width}')
        if type(self.y_start) is not int or not 0 <= self.y_start < y_end:
            raise ValueError('Expected 0 <= valid_y_start < the resolved valid y end')
        return slice(self.y_start, y_end), slice(self.x_start, x_end)

    def build_mask(self, image: Image, mask: Mask | None = None) -> Mask:
        """Return a new HxW uint8 image ROI, never annotation knowledge."""
        region = self.slices(image)
        if mask is not None and (not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]):
            raise ValueError('Valid-mask size must match image height and width')
        cropped = (np.ones(image[region].shape[:2], np.uint8) if mask is None
                   else (mask[region] > 0).astype(np.uint8))
        if cropped.sum() < 1024:
            raise ValueError('Too few valid image pixels')
        if cropped.shape == image.shape[:2]:
            return cropped
        out = np.zeros(image.shape[:2], np.uint8)
        out[region] = cropped
        return out

    def crop(self, image: Image) -> Image:
        """Return a copy of the valid rectangle."""
        return image[self.slices(image)].copy()

    @staticmethod
    def resize_for_matching(gray: Image, mask: Mask, size: Size) -> tuple[Image, Mask]:
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

    def describe(self, image: Image, mask: Mask) -> Diagnostics:
        """JSON-safe record of the resolved image region, including mask holes."""
        ys, xs = self.slices(image)
        info = dict(image_height=int(image.shape[0]), image_width=int(image.shape[1]),
                    valid_height=ys.stop, valid_pixels=int(mask.sum()),
                    height_source='explicit' if self.valid_height is not None else 'full_image')
        if (ys.start, ys.stop, xs.start, xs.stop) != (0, image.shape[0], 0, image.shape[1]):
            info['valid_bounds'] = dict(x_start=xs.start, x_end=xs.stop, y_start=ys.start, y_end=ys.stop)
        return info
