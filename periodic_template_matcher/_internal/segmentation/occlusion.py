"""Extend verified foreground where a defect hides the target's local structure."""

from itertools import product
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils import ValidRegion, to_gray, warp_labels
from ..models import MatchResult, PreparedInputs
from ..registration.core import _features
from .layers import projected_components, record_periodic_prediction
from .periodic import _candidate_fill

if TYPE_CHECKING:
    from ..objects.matching import ObjectPrediction


class LatticeOcclusionCompletionStep:
    """Use visible target structure to validate shifted template annotations."""

    def run(self, inputs: PreparedInputs, result: MatchResult, cfg: MatcherConfig,
            prediction: 'ObjectPrediction | None' = None) -> None:
        if not (cfg.robust_appearance and cfg.periodic_completion and result.direct_labels is not None
                and result.direct_valid is not None and result.label_periodic_valid is not None):
            return
        assert result.mapped_labels is not None and result.label_valid is not None
        info = result.registration.get('periodic_completion', {})
        axes = [axis for axis in info.get('lattice', []) if axis.get('verified')
                and axis.get('target_verification', {}).get('verified')]
        if not axes:
            return
        roi = inputs.target_roi
        semantic_missing = roi & ~result.direct_valid & (result.mapped_labels == 0)
        has_identities = (prediction is not None
                          and prediction.diagnostics.get('status') in ('accepted', 'partial_reference'))
        object_missing = roi & ~prediction.valid if has_identities else None
        if not semantic_missing.any() and (object_missing is None or not object_missing.any()):
            return

        h, w = roi.shape
        scale = cfg.matching_scale(result.template_image.shape, inputs.target.shape)
        size = (round(w*scale), round(h*scale))
        gray, valid = ValidRegion.resize_for_matching(to_gray(inputs.target), roi, size)
        detail = cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (cfg.occlusion_texture_window,)*2)
        tone = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), cfg.occlusion_tone_sigma)
        texture_floor = max(cfg.occlusion_texture_floor,
                            float(np.median(detail[valid > 0])) * cfg.occlusion_texture_floor_ratio)

        structure, _, _, support = _features(gray, valid, low_frequency=True, cfg=cfg)
        references = _registered_annotations(inputs, result) if semantic_missing.any() else []
        identities = prediction.identities.copy() if has_identities else None
        semantic_work = _resize_missing(semantic_missing, size)
        object_work = (_resize_missing(object_missing, size) if has_identities
                       else np.zeros_like(semantic_work))
        if not (semantic_work.any() or object_work.any()):
            return
        shifts = product(range(-cfg.periodic_max_steps, cfg.periodic_max_steps + 1), repeat=len(axes))
        shifts = sorted((steps for steps in shifts if any(steps)), key=lambda s: sum(map(abs, s)))
        records = []
        newly_covered = 0
        detected = np.zeros_like(roi)
        for steps in shifts:
            vectors = [axis['source_vector'] for n, axis in zip(steps, axes) if n]
            allow_objects = (object_missing is not None
                             and cfg.allows_periodic_shift(cfg.ordered_object_class, vectors))
            missing = ((semantic_work if references else False) | (object_work if allow_objects else False))
            if not np.any(missing):
                continue
            shift = np.asarray(steps) @ np.asarray([axis['target_vector'] for axis in axes])
            translation = np.eye(2, 3, dtype=np.float64)
            translation[:, 2] = shift * [size[0]/w, size[1]/h]
            hidden = _supported_occlusion(structure, support, valid, detail, tone,
                                          texture_floor, translation, missing, cfg)
            if not hidden.any():
                continue
            hidden = cv2.resize(hidden.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            detected |= hidden & roi
            translation[:, 2] = shift
            semantic_pixels = 0
            object_pixels = 0
            for index, labels, annotation, matrix, classes in references:
                if not (hidden & semantic_missing).any():
                    break
                inverse = np.linalg.inv(matrix[:2, :2])
                source_vectors = [inverse @ np.asarray(axis['target_vector'])
                                  for n, axis in zip(steps, axes) if n]
                allowed = [c for c in classes if c != cfg.ordered_object_class
                           and cfg.allows_periodic_shift(c, source_vectors)]
                if not allowed:
                    continue
                pose = matrix.copy()
                pose[:2] += np.outer(shift, pose[2])
                mapped, annotated = warp_labels(labels, pose, roi.shape, annotation)
                support_mask = annotated & roi & np.isin(mapped, allowed)
                fill = _candidate_fill(mapped, hidden & support_mask, roi, result.mapped_labels, ~semantic_missing)
                count = int(fill.sum())
                if count:
                    record_periodic_prediction(result, mapped, projected_components(mapped, support_mask, fill))
                    newly_covered += int(np.count_nonzero(fill & ~result.label_valid))
                    result.mapped_labels[fill] = mapped[fill]
                    result.label_valid |= fill
                    result.label_periodic_valid |= fill
                    semantic_missing &= ~fill
                    semantic_pixels += count
                    records.append(dict(template_index=index, target_vector=shift.tolist(),
                                        semantic_pixels=count, object_pixels=0))
            if semantic_pixels:
                semantic_work = _resize_missing(semantic_missing, size)
            if allow_objects and (hidden & object_missing).any():
                mapped = _shift_labels(identities, translation)
                fill = hidden & object_missing & (mapped > 0)
                fill = _connected_object_extension(fill, mapped, identities)
                prediction.labels[fill] = cfg.ordered_object_class
                prediction.identities[fill] = mapped[fill]
                prediction.valid |= fill
                object_missing &= ~fill
                object_pixels = int(fill.sum())
                if object_pixels:
                    object_work = _resize_missing(object_missing, size)
            if object_pixels:
                records.append(dict(target_vector=shift.tolist(), semantic_pixels=0,
                                    object_pixels=object_pixels))
        result.occlusion_mask = detected
        result.registration['occlusion'] = dict(
            candidate_pixels=int(detected.sum()), texture_floor=texture_floor,
            semantic_fill_pixels=sum(record['semantic_pixels'] for record in records),
            object_fill_pixels=sum(record['object_pixels'] for record in records))
        if records:
            missing = int((roi & ~result.label_valid).sum())
            info.update(occlusion_copies=records, remaining_unknown_pixels=missing,
                        label_coverage_roi=float(result.label_valid.sum()/roi.sum()),
                        added_label_pixels=int(result.label_periodic_valid.sum()),
                        status='incomplete' if missing else 'complete')
            info['newly_covered_pixels'] = info.get('newly_covered_pixels', 0) + newly_covered
            inferred_background = sum(record['semantic_pixels'] for record in records) - newly_covered
            info['boundary_foreground_pixels'] = info.get('boundary_foreground_pixels', 0) + inferred_background


def _registered_annotations(inputs, result):
    references = []
    for index in result.registration['ranking']:
        registration = result.registration['matches'][index]
        pose = registration.get('matrix_template_to_target')
        labels = inputs.labels[index]
        if not registration.get('geometry_accepted') or pose is None or labels is None:
            continue
        annotation = inputs.template_valids[index]
        if inputs.annotation_valids[index] is not None:
            annotation = annotation & (inputs.annotation_valids[index] > 0)
        classes = [int(c) for c in np.unique(labels) if c > 0]
        references.append((index, labels, annotation, np.asarray(pose, float), classes))
    return references


def _resize_missing(missing, size):
    return cv2.resize(missing.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0


def _shift_labels(labels, translation):
    return cv2.warpAffine(labels, translation, labels.shape[::-1], flags=cv2.INTER_NEAREST)


def _connected_object_extension(fill, mapped, confirmed):
    """Keep only extensions that join an existing contour of the same identity."""
    approved = np.zeros_like(fill)
    height, width = fill.shape
    for identity in np.unique(mapped[fill]):
        proposed = (fill & (mapped == identity)).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(proposed)
        roi = np.s_[max(0, y-1):min(height, y+h+1), max(0, x-1):min(width, x+w+1)]
        _, components = cv2.connectedComponents(proposed[roi])
        contact = cv2.dilate((confirmed[roi] == identity).astype(np.uint8), np.ones((3, 3), np.uint8))
        attached = np.unique(components[(contact > 0) & (proposed[roi] > 0)])
        approved[roi] |= np.isin(components, attached)
    return approved


def _large_regions(mask, support, *, diameter, min_area):
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))) & support
    _, components, stats, _ = cv2.connectedComponentsWithStats(closed)
    large = stats[:, cv2.CC_STAT_AREA] >= min_area
    large[0] = False
    return large[components]


def _supported_occlusion(structure, support, valid, detail, tone, texture_floor, translation, missing, cfg):
    """Find suppressed texture near independent, strongly matching peer structure."""
    size = structure.shape[::-1]
    peer = cv2.warpAffine(structure, translation, size)
    peer_detail = cv2.warpAffine(detail, translation, size)
    peer_tone = cv2.warpAffine(tone, translation, size)
    common = valid & cv2.warpAffine(valid, translation, size, flags=cv2.INTER_NEAREST)
    loss = (common > 0) & (peer_detail >= texture_floor) & (detail < peer_detail*cfg.occlusion_texture_ratio)
    # Dark occluders can have their own edges. Compare low-frequency brightness
    # with the visible peer, and require a coherent deficit much larger than a
    # normal local fluctuation between nominal repeats.
    dark = (common > 0) & (tone < peer_tone*cfg.occlusion_dark_ratio) & (peer_tone-tone >= cfg.occlusion_dark_delta)
    if not ((loss | dark) & missing).any():
        return np.zeros_like(loss)
    # Suppression is a region property. Close small texture islands inside a
    # defect but discard isolated low-contrast patches in otherwise sharp cells.
    obscured = _large_regions(loss, common, diameter=cfg.occlusion_texture_close, min_area=cfg.occlusion_texture_area)
    if (dark & missing).any():
        dark_core = _large_regions(dark, common, diameter=cfg.occlusion_dark_close, min_area=cfg.occlusion_dark_area)
        # Close holes left by defect texture inside one coherent dark region.
        if dark_core.any():
            envelope = _large_regions(dark_core | obscured, common, diameter=cfg.occlusion_envelope_close,
                                      min_area=cfg.occlusion_dark_area)
            _, components = cv2.connectedComponents(envelope.astype(np.uint8))
            dark_components = np.unique(components[dark_core & envelope])
            obscured |= np.isin(components, dark_components)
    fill = obscured & missing
    if not fill.any():
        return fill

    def average(values):
        return cv2.blur(values, (cfg.occlusion_anchor_window,)*2, borderType=cv2.BORDER_CONSTANT)

    # A can reach the real image edge. Verify its position in the larger
    # neighborhood B, using only safe visible structure on both sides.
    safe = support & cv2.warpAffine(support, translation, size, flags=cv2.INTER_NEAREST)
    visible = ((safe > 0) & ~loss & ~dark).astype(np.float32)
    fraction = average(visible)
    count = np.maximum(fraction, 1e-6)
    a, b = structure*visible, peer*visible
    ma, mb = average(a)/count, average(b)/count
    va, vb = average(a*a)/count-ma*ma, average(b*b)/count-mb*mb
    covariance = average(a*b)/count-ma*mb
    anchors = ((fraction >= cfg.occlusion_anchor_visible)
               & (va > cfg.occlusion_anchor_variance) & (vb > cfg.occlusion_anchor_variance)
               & (covariance >= cfg.occlusion_anchor_ncc*np.sqrt(np.maximum(va*vb, 0.))))
    if not anchors.any():
        return np.zeros_like(loss)
    distance = cv2.distanceTransform((~anchors).astype(np.uint8), cv2.DIST_L2, 5)
    # Bound the extension to half a verified lattice displacement.
    return obscured & (distance <= np.linalg.norm(translation[:, 2])*cfg.occlusion_anchor_max_period_ratio)
