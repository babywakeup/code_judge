"""Deterministic, array-only SAM3 prompt planning; no inference or annotation IO.

Coordinates are zero-based, half-open xyxy bounds and integer xy pixel centers.
Anomaly mode requires all three evidence arrays (including confidence); missing
or malformed evidence raises ValueError rather than inventing reliable reference.
Box mode deliberately ignores evidence, even when supplied and malformed.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .box_graphcut import DetectionBox
from .sam3_guidance import GUIDANCE_CONFIG, guidance_evidence


# Fixed algorithm choices, suitable for copying into prediction provenance.
CONFIG = dict(
    context_fraction=0.1,
    min_context=16,
    foreground_threshold=3.0,
    background_threshold=1.0,
    reliable_confidence=255,
    confidence_values=(0, 128, 255),
    connectivity=8,
    max_components=16,
    positive_points_per_component=3,
    background_points=8,
    interior_fraction=0.5,
    distance_transform_mask=5,
    component_order='peak_descending_then_id',
    point_order='deepest_then_farthest_interior_then_depth_then_row_major',
)


def _spread(mask: np.ndarray, limit: int) -> list[list[int]]:
    """Spread points through the interior; ties resolve in row-major order.

    Zero padding makes image/ROI edges real boundaries, including full masks.
    The interior is the deepest half of the distance-transform range (at least
    one pixel). Thin components retain all their pixels as candidates.
    """
    depth = cv2.distanceTransform(
        np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2,
        CONFIG['distance_transform_mask'],
    )[1:-1, 1:-1]
    threshold = max(1.0, float(depth.max()) * CONFIG['interior_fraction'])
    ys, xs = np.nonzero(mask & (depth >= threshold))
    if not len(xs):
        return []
    depths = depth[ys, xs]
    chosen = int(np.argmax(depths))
    nearest = np.full(len(xs), np.inf)
    available = np.ones(len(xs), bool)
    points = []
    for _ in range(min(limit, len(xs))):
        points.append([int(xs[chosen]), int(ys[chosen])])
        available[chosen] = False
        distance = ((xs.astype(np.float64) - xs[chosen]) ** 2
                    + (ys.astype(np.float64) - ys[chosen]) ** 2)
        np.minimum(nearest, distance, out=nearest)
        candidates = np.flatnonzero(available)
        if not len(candidates):
            break
        farthest = candidates[nearest[candidates] == nearest[candidates].max()]
        chosen = int(farthest[np.argmax(depths[farthest])])
    return points


def _evidence(image_shape, score, valid, confidence):
    if (not isinstance(score, np.ndarray) or score.shape != image_shape
            or score.dtype.kind not in 'fiu' or not np.isfinite(score).all()
            or np.any(score < 0)):
        raise ValueError('score must be an input-sized finite nonnegative numeric array')
    if (not isinstance(valid, np.ndarray) or valid.shape != image_shape
            or valid.dtype != np.bool_):
        raise ValueError('valid must be an input-sized bool array')
    if (not isinstance(confidence, np.ndarray) or confidence.shape != image_shape
            or confidence.dtype != np.uint8
            or not np.isin(confidence, CONFIG['confidence_values']).all()):
        raise ValueError('confidence must be an input-sized uint8 array containing 0/128/255')
    if np.any(valid != (confidence != 0)):
        raise ValueError('valid must equal confidence > 0 (128 is valid but not reliable)')
    return valid & (confidence == CONFIG['reliable_confidence'])


def plan_prompts(image_shape: tuple[int, int], boxes: list[DetectionBox], *,
                 mode='anomaly_points', score: np.ndarray | None = None,
                 valid: np.ndarray | None = None,
                 confidence: np.ndarray | None = None,
                 context_fraction: float = CONFIG['context_fraction']) -> list[dict]:
    """Return one record per detector box, with one task per selected component.

    All task coordinates are local to the clipped padded ROI. Positive points
    precede shared background negatives. Component IDs are OpenCV's 8-connected
    labels within that detection's crop, not IDs shared between detections.
    Missing reference/foreground/background pixels explicitly abstain; invalid
    arguments raise ValueError, including for empty/out-of-frame detections.
    context_fraction expands each crop side by that fraction of the ORIGINAL
    box width/height (at least min_context pixels), never the prompt box itself.
    """
    if (not isinstance(image_shape, tuple) or len(image_shape) != 2
            or any(type(n) is not int or n <= 0 for n in image_shape)):
        raise ValueError('image_shape must be a positive integer (height, width) tuple')
    if not isinstance(boxes, list):
        raise ValueError('boxes must be a list of DetectionBox instances')
    for box in boxes:
        if not isinstance(box, DetectionBox):
            raise ValueError('boxes must contain DetectionBox instances')
        box.validate()
    if mode not in ('box', 'tracker_box', 'anomaly_points', 'anomaly_guided'):
        raise ValueError('mode must be box, tracker_box, anomaly_points or anomaly_guided')
    if (isinstance(context_fraction, bool) or not isinstance(context_fraction, (int, float))
            or not math.isfinite(context_fraction) or context_fraction < 0):
        raise ValueError('context_fraction must be a finite nonnegative number')
    reliable = None
    if mode in ('anomaly_points', 'anomaly_guided'):
        reliable = _evidence(image_shape, score, valid, confidence)

    height, width = image_shape
    detection_union = None
    if mode in ('anomaly_points', 'anomaly_guided'):
        detection_union = np.zeros(image_shape, bool)
        for box in boxes:
            detection_union[max(0, box.y0):max(0, min(height, box.y1)),
                            max(0, box.x0):max(0, min(width, box.x1))] = True

    records = []
    for index, box in enumerate(boxes):
        record = dict(index=index, box_xyxy=[box.x0, box.y0, box.x1, box.y1],
                      roi_xyxy=None, status='outside_image', warnings=[], prompts=[])
        records.append(record)
        bx0, by0 = max(0, box.x0), max(0, box.y0)
        bx1, by1 = min(width, box.x1), min(height, box.y1)
        if bx0 >= bx1 or by0 >= by1:
            continue
        # Clip padding before multiplication: even huge finite fractions should
        # saturate at the image boundary rather than overflow to infinity.
        dx = max(CONFIG['min_context'], math.ceil(
            (box.x1 - box.x0) * min(context_fraction, width / (box.x1 - box.x0))))
        dy = max(CONFIG['min_context'], math.ceil(
            (box.y1 - box.y0) * min(context_fraction, height / (box.y1 - box.y0))))
        x0, y0 = max(0, box.x0 - dx), max(0, box.y0 - dy)
        x1, y1 = min(width, box.x1 + dx), min(height, box.y1 + dy)
        record['roi_xyxy'] = [x0, y0, x1, y1]
        local_box = [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0]
        if mode in ('box', 'tracker_box'):
            record.update(status='ok', prompts=[dict(
                box_xyxy=local_box, points_xy=[], point_labels=[], component_id=None)])
            continue

        # Evidence has been validated globally, before processing any detection.
        assert reliable is not None and score is not None and detection_union is not None
        roi = np.s_[y0:y1, x0:x1]
        trusted, response = reliable[roi], score[roi]
        inside = np.zeros(trusted.shape, bool)
        lx0, ly0, lx1, ly1 = local_box
        inside[ly0:ly1, lx0:lx1] = True
        if not np.any(trusted & inside):
            record['status'] = 'no_reliable_reference'
            continue
        if mode == 'anomaly_guided':
            evidence = guidance_evidence(response, trusted, inside)
            if not evidence.component_ids:
                record['status'] = 'no_foreground_seeds'
                continue
            if evidence.component_count > len(evidence.component_ids):
                record['warnings'].append('seed_components_truncated')
            inside_negatives = _spread(evidence.near_normal & inside,
                                       GUIDANCE_CONFIG['inside_negative_points'])
            outside_negatives = _spread(evidence.near_normal & ~detection_union[roi],
                                        GUIDANCE_CONFIG['outside_negative_points'])
            if not inside_negatives:
                record['warnings'].append('no_near_normal_inside_box')
            if not outside_negatives:
                record['warnings'].append('no_near_normal_outside_boxes')
            negatives = inside_negatives + outside_negatives
            if not negatives:
                record['status'] = 'no_background_seeds'
                continue
            positives = [p for group in evidence.component_ids for p in _spread(
                evidence.core & (evidence.components == group),
                GUIDANCE_CONFIG['positive_points_per_component'])]
            record.update(status='ok', prompts=[dict(
                box_xyxy=local_box, points_xy=positives + negatives,
                point_labels=[1] * len(positives) + [0] * len(negatives),
                component_id=None, component_ids=evidence.component_ids,
                inside_negative_count=len(inside_negatives), outside_negative_count=len(outside_negatives))])
            continue
        count, components = cv2.connectedComponents(
            (inside & trusted & (response >= CONFIG['foreground_threshold'])).astype(np.uint8),
            connectivity=CONFIG['connectivity'],
        )
        if count == 1:
            record['status'] = 'no_foreground_seeds'
            continue
        # Linear scan for component peaks avoids a whole-ROI scan per component.
        peaks = np.zeros(count, dtype=response.dtype)
        np.maximum.at(peaks, components.ravel(), response.ravel())
        groups = sorted(range(1, count), key=lambda k: (-peaks[k].item(), k))
        if len(groups) > CONFIG['max_components']:
            record['warnings'].append('seed_components_truncated')
        groups = groups[:CONFIG['max_components']]
        background = trusted & ~detection_union[roi] & (response <= CONFIG['background_threshold'])
        if not background.any():
            record['status'] = 'no_background_seeds'
            continue
        negatives = _spread(background, CONFIG['background_points'])
        for group in groups:
            positives = _spread(components == group, CONFIG['positive_points_per_component'])
            record['prompts'].append(dict(
                box_xyxy=local_box.copy(),
                points_xy=positives + [p.copy() for p in negatives],
                point_labels=[1] * len(positives) + [0] * len(negatives),
                component_id=int(group),
            ))
        record['status'] = 'ok'
    return records