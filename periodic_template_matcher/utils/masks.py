"""Mask validation, template-to-target label warping and label overlays."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ..config import MatcherConfig
from .regions import ValidRegion
from .types import BoolMask, ClassIds, Image, LabelInput, LabelMap, Mask, Matrix, Shape


def build_valid_mask(gray: Image, mask: Mask | None, cfg: MatcherConfig,
                     valid_height: int | None = None) -> Mask:
    """Build a matching ROI; label knowledge is represented separately."""
    return ValidRegion.from_config(cfg, valid_height).build_mask(gray, mask)


def normalize_masks(masks: LabelInput, shape: Shape, name: str = 'masks') -> LabelMap:
    """Accept an HxW class-ID array or disjoint binary masks for classes 1..K.

    An empty binary-mask list means known background everywhere. A class-ID
    array retains its IDs (including 255); binary list entries use 0/1 or 0/255.
    """
    if isinstance(masks, np.ndarray):
        if masks.shape != tuple(shape) or masks.ndim != 2:
            raise ValueError(f'{name} must match image height and width')
        if masks.dtype != np.bool_ and not np.issubdtype(masks.dtype, np.integer):
            raise ValueError(f'{name} must contain integer class IDs')
        if masks.size == 0 or masks.min() < 0 or masks.max() > 65535:
            raise ValueError(f'{name} class IDs must be in 0..65535')
        return masks.astype(np.uint16, copy=True)
    if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
        raise ValueError(f'{name} must be a class-ID array or a list of binary masks')
    if len(masks) > 65535:
        raise ValueError(f'{name} supports at most 65535 binary classes')
    labels = np.zeros(shape, np.uint16)
    for class_id, mask in enumerate(masks, 1):
        if not isinstance(mask, np.ndarray) or mask.shape != tuple(shape):
            raise ValueError(f'{name}[{class_id - 1}] must match image height and width')
        if mask.dtype != np.bool_ and not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f'{name} binary masks must be boolean or integer')
        if not np.isin(mask, (0, 1)).all() and not np.isin(mask, (0, 255)).all():
            raise ValueError(f'{name} binary masks must use 0/1 or 0/255')
        foreground = mask > 0
        if np.any(foreground & (labels > 0)):
            raise ValueError(f'{name} binary class masks overlap; use exclusive classes')
        labels[foreground] = class_id
    return labels


def warp_valid_mask(mask: Mask, H: Matrix, target_shape: Sequence[int]) -> BoolMask:
    """Project an HxW validity mask to target coordinates with nearest neighbors."""
    if mask.ndim != 2:
        raise ValueError('valid mask must be a 2D array')
    H = np.asarray(H, np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10:
        raise ValueError('H must be an invertible 3x3 TEMPLATE-to-TARGET matrix')
    h, w = target_shape[:2]
    return cv2.warpPerspective((mask > 0).astype(np.uint8), H, (w, h),
                               flags=cv2.INTER_NEAREST) > 0


def warp_labels(labels: ClassIds, H: Matrix, target_shape: Sequence[int],
                source_valid: Mask | None = None) -> tuple[LabelMap, BoolMask]:
    """Warp integer IDs TEMPLATE -> TARGET; False validity means unknown.

    Returns uint16 labels and boolean validity. No periodic extrapolation.
    Label 0 can be true background; known pixels must be tracked separately.
    """
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('labels must be a 2D integer class-ID array')
    if labels.min() < 0 or labels.max() > 65535:
        raise ValueError('Use nonnegative class IDs <= 65535')
    if source_valid is not None and source_valid.shape != labels.shape:
        raise ValueError('source_valid must match labels')
    H = np.asarray(H, np.float64)
    source_mask = np.ones(labels.shape, np.uint8) if source_valid is None else source_valid
    valid = warp_valid_mask(source_mask, H, target_shape)
    h, w = target_shape[:2]
    warped = cv2.warpPerspective(labels.astype(np.uint16), H, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, valid


def overlay_labels(image: Image, labels: ClassIds, valid: BoolMask) -> Image:
    """Blend non-background class colors without mutating input arrays."""
    image = image.copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    selected = valid & (labels > 0)
    colors = np.stack([(labels.astype(np.uint32)*67+39)%256,
                       (labels.astype(np.uint32)*131+83)%256,
                       (labels.astype(np.uint32)*197+127)%256], axis=-1).astype(np.uint8)
    image[selected] = (0.55*image[selected]+0.45*colors[selected]).astype(np.uint8)
    return image
