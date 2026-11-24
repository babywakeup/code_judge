"""Compact connected-region masks with exact full-frame nearest sampling."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskRegion:
    """A binary region's position and bounded pixels, not a target-sized copy."""

    y: int
    x: int
    mask: np.ndarray

    @property
    def slices(self):
        return (slice(self.y, self.y + self.mask.shape[0]),
                slice(self.x, self.x + self.mask.shape[1]))

    @property
    def pixels(self):
        return int(self.mask.sum())


def component_regions(mask):
    """Keep OpenCV's exact component IDs/order/connectivity and tight bounds."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    for component in range(1, count):
        x, y, width, height = stats[component, :4]
        yield MaskRegion(int(y), int(x), components[y:y + height, x:x + width] == component)


class RegionResize:
    """Resize a compact mask exactly as cv2.resize on its full zero canvas.

    Cropping then resizing changes pixel-center alignment. Derive source sample
    positions with OpenCV itself, once, and only gather samples inside the box.
    Context dilation and all scoring still use the original working canvas.
    """

    def __init__(self, shape, size):
        self.size = size
        self.x = cv2.resize(np.arange(shape[1], dtype=np.float64)[None, :],
                            (size[0], 1), interpolation=cv2.INTER_NEAREST)[0].astype(np.intp)
        self.y = cv2.resize(np.arange(shape[0], dtype=np.float64)[:, None],
                            (1, size[1]), interpolation=cv2.INTER_NEAREST)[:, 0].astype(np.intp)

    def __call__(self, region: MaskRegion):
        output = np.zeros((self.size[1], self.size[0]), np.uint8)
        x0, x1 = np.searchsorted(self.x, [region.x, region.x + region.mask.shape[1]])
        y0, y1 = np.searchsorted(self.y, [region.y, region.y + region.mask.shape[0]])
        if x1 > x0 and y1 > y0:
            output[y0:y1, x0:x1] = region.mask[np.ix_(self.y[y0:y1] - region.y,
                                                     self.x[x0:x1] - region.x)]
        return output