"""Full-resolution multi-window SSIM, not standard downsampled MS-SSIM."""
from __future__ import annotations

import cv2
import numpy as np

from periodic_template_matcher.utils import to_gray


def multiscale_ssim(image: np.ndarray, reference: np.ndarray, valid: np.ndarray,
                    source_map: np.ndarray | None = None) -> tuple:
    """Return max(1-SSIM), comparable mask and separate 5/11/21-pixel maps.

    Only complete windows inside one valid source fragment are comparable.
    Luminance, contrast and covariance use fixed uint8 dynamic range constants.
    """
    a, b = to_gray(image).astype(np.float32), to_gray(reference).astype(np.float32)
    response = np.zeros(valid.shape, np.float32)
    known = np.zeros(valid.shape, bool)
    maps = {}
    ids = None
    if source_map is not None:
        # Compact arbitrary integer IDs before float morphology. The ranks are
        # exact even when original uint64 IDs exceed float64 integer precision.
        _, compact = np.unique(source_map, return_inverse=True)
        ids = compact.reshape(valid.shape).astype(np.float64)
    for size in (5, 11, 21):
        kernel = np.ones((size, size), np.uint8)
        window_valid = cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if source_map is not None:
            window_valid &= cv2.erode(ids, kernel) == cv2.dilate(ids, kernel)
        sigma = size/6
        def blur(x):
            return cv2.GaussianBlur(x, (size, size), sigma)
        ma, mb = blur(a), blur(b)
        va, vb = np.maximum(0, blur(a*a)-ma*ma), np.maximum(0, blur(b*b)-mb*mb)
        covariance = blur(a*b)-ma*mb
        luminance = (2*ma*mb+6.5025)/(ma*ma+mb*mb+6.5025)
        structure = (2*covariance+58.5225)/(va+vb+58.5225)
        residual = np.clip(1-luminance*structure, 0, 2).astype(np.float32)
        residual[~window_valid] = 0
        maps[size] = residual
        np.maximum(response, residual, out=response)
        known |= window_valid
    return response, known, maps
