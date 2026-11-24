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
    object_identities: LabelMap | None = None
    direct_labels: LabelMap | None = None
    direct_valid: BoolMask | None = None
    completion_labels: LabelMap | None = None
    completion_valid: BoolMask | None = None
    object_labels: LabelMap | None = None
    object_layer_valid: BoolMask | None = None
    object_layer_identities: LabelMap | None = None

    @property
    def class_ids(self) -> list[int]:
        if self.labels is None or self.valid is None:
            return []
        values = set(int(value) for value in np.unique(self.labels[self.valid]))
        for labels, valid in ((self.direct_labels, self.direct_valid),
                              (self.completion_labels, self.completion_valid),
                              (self.object_labels, self.object_layer_valid)):
            if labels is not None and valid is not None:
                values.update(int(value) for value in np.unique(labels[valid]))
        return sorted(values)


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
    object_identities = None
    if 'object_identities' in artifacts:
        if labels is None:
            raise ValueError('Object identity artifacts require an accepted prediction')
        object_identities = read_image(artifact('object_identities'), cv2.IMREAD_UNCHANGED)
        if object_identities.shape != shape or object_identities.dtype != np.uint16:
            raise ValueError('Object identity dimensions/type must match input')
    layers = {}
    for layer in ('direct', 'completion', 'object'):
        label_key = layer+'_labels'
        valid_key = 'object_layer_valid' if layer == 'object' else layer+'_valid'
        if label_key not in artifacts and valid_key not in artifacts:
            continue
        if labels is None:
            raise ValueError('Independent layers require an accepted prediction')
        layer_labels = read_image(artifact(label_key), cv2.IMREAD_UNCHANGED)
        layer_valid = read_image(artifact(valid_key), cv2.IMREAD_UNCHANGED)
        if (layer_labels.shape != shape or layer_valid.shape != shape
                or layer_labels.dtype != np.uint16 or layer_valid.dtype != np.uint8):
            raise ValueError(f'{layer} layer dimensions/type must match input')
        layers[label_key] = layer_labels
        layers[valid_key] = layer_valid > 0
    if 'object_layer_identities' in artifacts:
        identities = read_image(artifact('object_layer_identities'), cv2.IMREAD_UNCHANGED)
        if 'object_labels' not in layers or identities.shape != shape or identities.dtype != np.uint16:
            raise ValueError('Independent object identities require matching uint16 object labels')
        layers['object_layer_identities'] = identities
    return SavedPrediction(manifest, metadata, image, labels, valid, periodic, object_identities, **layers)


def _render_layers(data, index, inside, source, class_id, object_identity):
    """Return independent evidence without forcing overlaps into one class ID."""
    if source == 'fused':
        return None
    layers = (('direct', data.direct_labels, data.direct_valid),
              ('periodic', data.completion_labels, data.completion_valid),
              ('objects', data.object_labels, data.object_layer_valid))
    if not any(labels is not None for _, labels, _ in layers):
        return None  # Older exports retain their exact historical display.
    selected = np.zeros(inside.shape, bool)
    valid = np.zeros(inside.shape, bool)
    color_sum = np.zeros((*inside.shape, 3), np.uint16)
    counts = np.zeros(inside.shape, np.uint8)
    seen = []
    for name, layer_labels, layer_valid in layers:
        if source not in ('all', name) or layer_labels is None or layer_valid is None:
            continue
        labels = layer_labels[index]
        known = layer_valid[index] & inside
        valid |= known
        pick = known & ((labels > 0) if class_id is None else labels == class_id)
        if object_identity is not None:
            if name != 'objects' or data.object_layer_identities is None:
                continue
            pick &= data.object_layer_identities[index] == object_identity
        # Repeated evidence for one class must not bias the overlap color.
        for earlier_labels, earlier_pick in seen:
            pick &= ~(earlier_pick & (earlier_labels == labels))
        color_sum[pick] += CLASS_COLORS[labels[pick]]
        counts[pick] += 1
        selected |= pick
        seen.append((labels, pick.copy()))
    colors = (color_sum / np.maximum(counts[..., None], 1)).astype(np.uint8)
    return colors, selected, valid


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
                show_unknown: bool = True, object_identity: int | None = None) -> Image:
    """Render the visible viewport, keeping labels/validity nearest-neighbor.

    origin is the input image's top-left position in viewport pixels. Side-by-side
    panels use the same transform. Rendering cost follows viewport size, so zoom
    never allocates a full enlarged image.
    """
    if mode not in ('input', 'mask', 'overlay', 'split') or source not in ('all', 'fused', 'direct', 'periodic', 'objects'):
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
        object_ids = np.zeros_like(labels) if data.object_identities is None else data.object_identities[index]
        if object_identity is not None:
            selected &= object_ids == object_identity
        if source not in ('all', 'fused'):
            periodic = np.zeros_like(valid)
            if data.periodic is not None:
                periodic = data.periodic[index] & inside
            selected &= (periodic if source == 'periodic' else (object_ids > 0) if source == 'objects'
                         else (~periodic & (object_ids == 0)))
        colors = CLASS_COLORS[labels]
        independent = _render_layers(data, index, inside, source, class_id, object_identity)
        if independent is not None:
            colors, selected, valid = independent
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
