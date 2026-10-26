"""Load saved prediction arrays and render viewport-sized inspection layers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np

from ..utils import read_image, validate_image
from ..utils.types import BoolMask, Diagnostics, Image, LabelMap, PathInput, Size


def _class_palette() -> Image:
    """Hash all 16 ID bits into readable BGR colors once, outside redraws."""
    ids = np.arange(65536, dtype=np.uint32)
    mixed = (ids ^ (ids >> 8)) * np.uint32(0x45d9f3b)
    mixed ^= mixed >> 16
    channels = np.stack([mixed & 255, (mixed >> 8) & 255, (mixed >> 16) & 255], axis=-1)
    return (64 + channels * 191 // 255).astype(np.uint8)


CLASS_COLORS = _class_palette()


@dataclass(frozen=True)
class SavedPrediction:
    """Original-sized input plus optional class labels and label knowledge."""

    manifest: Path
    metadata: Diagnostics
    image: Image
    labels: LabelMap | None
    valid: BoolMask | None
    periodic: BoolMask | None

    @property
    def class_ids(self) -> list[int]:
        if self.labels is None or self.valid is None:
            return []
        return [int(value) for value in np.unique(self.labels[self.valid])]


def load_result(path: PathInput) -> SavedPrediction:
    """Open a stage-1 directory or manifest, without access to original inputs."""
    manifest = Path(path).resolve()
    if manifest.is_dir():
        manifest /= 'result.json'
    metadata = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if not isinstance(metadata, dict) or metadata.get('format_version') != 1 or not isinstance(metadata.get('artifacts'), dict):
        raise ValueError('Unsupported result format; rerun inference with save_result().')
    matches, ranking = metadata.get('matches'), metadata.get('ranking')
    if (not isinstance(matches, list) or not isinstance(ranking, list) or not ranking
            or type(ranking[0]) is not int or not 0 <= ranking[0] < len(matches)
            or not isinstance(matches[ranking[0]], dict)
            or not isinstance(matches[ranking[0]].get('match_score'), (int, float))):
        raise ValueError('Invalid match ranking/score in result.json')
    if metadata.get('prediction_status') not in ('available', 'match_rejected', 'no_template_annotations'):
        raise ValueError('Invalid prediction status in result.json')
    artifacts = metadata['artifacts']

    def artifact(key: str) -> Path:
        name = artifacts.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f'Missing result artifact: {key}')
        resolved = (manifest.parent / name).resolve()
        if not resolved.is_relative_to(manifest.parent):
            raise ValueError(f'Artifact must be inside the result directory: {key}')
        return resolved

    image = read_image(artifact('input'), cv2.IMREAD_UNCHANGED)
    validate_image(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    shape = image.shape[:2]
    labels = valid = periodic = None
    if metadata.get('prediction_status') == 'available':
        if not metadata.get('accepted'):
            raise ValueError('Rejected result cannot contain an accepted prediction.')
        labels = read_image(artifact('prediction_mask'), cv2.IMREAD_UNCHANGED)
        raw_valid = read_image(artifact('prediction_valid'), cv2.IMREAD_UNCHANGED)
        if labels.shape != shape or raw_valid.shape != shape:
            raise ValueError('Prediction mask/validity dimensions must match input image size.')
        if labels.dtype != np.uint16 or raw_valid.dtype != np.uint8:
            raise ValueError('Expected uint16 class IDs and uint8 label validity.')
        valid = raw_valid > 0
        if 'periodic_valid' in artifacts:
            raw_periodic = read_image(artifact('periodic_valid'), cv2.IMREAD_UNCHANGED)
            if raw_periodic.shape != shape or raw_periodic.dtype != np.uint8:
                raise ValueError('Periodic validity dimensions/type must match input image.')
            periodic = (raw_periodic > 0) & valid
    return SavedPrediction(manifest, metadata, image, labels, valid, periodic)


def class_color(class_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a class ID, including uint16 IDs."""
    return tuple(int(channel) for channel in CLASS_COLORS[class_id])


def source_pixel(x: int, y: int, scale: float,
                  origin: tuple[float, float]) -> tuple[int, int]:
    """Map a viewport pixel to the exact nearest label pixel used by rendering."""
    return (int(np.floor((x - origin[0]) / scale + .5)),
            int(np.floor((y - origin[1]) / scale + .5)))


def render_view(data: SavedPrediction, size: Size, scale: float,
                origin: tuple[float, float], *, mode: str = 'overlay', alpha: float = .5,
                class_id: int | None = None, source: str = 'all',
                show_unknown: bool = True) -> Image:
    """Render the visible viewport, keeping labels/validity nearest-neighbor.

    origin is the input image's top-left position in viewport pixels. Side-by-side
    panels use the same transform. Rendering cost follows viewport size, so zoom
    never allocates a full enlarged image.
    """
    if mode not in ('input', 'mask', 'overlay', 'split') or source not in ('all', 'direct', 'periodic'):
        raise ValueError('Invalid display mode or prediction source')
    if not np.isfinite(scale) or scale <= 0 or min(size) < 1 or not 0 <= alpha <= 1:
        raise ValueError('Invalid viewport scale, size or opacity')
    width, height = size
    pane_width = max(1, width // 2) if mode == 'split' else width
    transform = np.array([[scale, 0., origin[0]], [0., scale, origin[1]]], np.float64)
    patch = cv2.warpAffine(data.image, transform, (pane_width, height),
                           flags=cv2.INTER_LINEAR, borderValue=(38, 30, 25))
    output = patch.copy()
    if mode != 'input' and data.labels is not None and data.valid is not None:
        # Explicit nearest sampling is shared with hover queries, including ties.
        ix = np.floor((np.arange(pane_width) - origin[0]) / scale + .5).astype(np.int64)
        iy = np.floor((np.arange(height) - origin[1]) / scale + .5).astype(np.int64)
        inside = ((iy[:, None] >= 0) & (iy[:, None] < data.image.shape[0]) &
                  (ix[None, :] >= 0) & (ix[None, :] < data.image.shape[1]))
        index = np.ix_(iy.clip(0, data.image.shape[0] - 1), ix.clip(0, data.image.shape[1] - 1))
        labels = data.labels[index]
        valid = data.valid[index] & inside
        selected = valid & ((labels > 0) if class_id is None else (labels == class_id))
        if source != 'all':
            periodic = np.zeros_like(valid)
            if data.periodic is not None:
                periodic = data.periodic[index] & inside
            selected &= periodic if source == 'periodic' else ~periodic
        colors = CLASS_COLORS[labels]
        if mode == 'mask':
            output[:] = (35, 40, 48)
            output[selected] = colors[selected]
        else:
            output[selected] = ((1 - alpha) * patch[selected] + alpha * colors[selected]).astype(np.uint8)
        if show_unknown:
            yy, xx = np.indices((height, pane_width))
            unknown = inside & ~valid
            hatch = ((xx + yy) % 12) < 3
            output[unknown & hatch] = (180, 95, 180)
            output[unknown & ~hatch] = (output[unknown & ~hatch] * .5).astype(np.uint8)
    if mode == 'split':
        canvas = np.full((height, width, 3), (38, 30, 25), np.uint8)
        canvas[:, :pane_width] = patch
        canvas[:, pane_width:2 * pane_width] = output
        canvas[:, pane_width:pane_width + 1] = (110, 120, 135)
        return canvas
    return output
