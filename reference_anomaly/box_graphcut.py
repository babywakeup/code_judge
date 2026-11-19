"""Array-only box/response graph-cut baseline. No annotation or GT access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow


@dataclass(frozen=True)
class DetectionBox:
    """Integer pixel bounds in zero-based half-open coordinates, unclipped."""
    x0: int
    y0: int
    x1: int
    y1: int
    label: str = 'defect'

    def validate(self):
        if (any(type(v) is not int for v in (self.x0, self.y0, self.x1, self.y1))
                or self.x1 <= self.x0 or self.y1 <= self.y0):
            raise ValueError('Boxes need increasing integer half-open bounds')


@dataclass(frozen=True)
class BoxGraphCutConfig:
    context_fraction: float = .1
    min_context: int = 16
    seed_threshold: float = 3.
    background_threshold: float = 1.
    max_components: int = 16
    seeds_per_component: int = 32
    anomaly_midpoint: float = 1.5
    anomaly_scale: float = .75
    appearance_weight: float = .6
    residual_scale_floor: float = 8.
    outside_penalty: float = .75
    smoothness: float = 2.5
    contrast_scale: float = 15.
    max_roi_pixels: int = 1_500_000

    def validate(self):
        values = asdict(self)
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for v in values.values()):
            raise ValueError('Graph-cut parameters must be finite numbers')
        for key in ('min_context', 'max_components', 'seeds_per_component', 'max_roi_pixels'):
            if type(values[key]) is not int or values[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if (self.context_fraction < 0 or self.smoothness < 0 or self.outside_penalty < 0
                or self.background_threshold < 0 or self.seed_threshold <= self.background_threshold
                or self.anomaly_scale <= 0 or self.contrast_scale <= 0
                or self.residual_scale_floor <= 0 or not 0 <= self.appearance_weight <= 1):
            raise ValueError('Invalid graph-cut thresholds, weights or scales')


@dataclass
class BoxPrediction:
    mask: np.ndarray
    valid: np.ndarray
    foreground_seeds: np.ndarray
    background_seeds: np.ndarray
    boxes: list[dict]
    status: str
    config: dict


def read_boxes(path: str | Path, image_shape: tuple[int, int], *,
               coordinates: str = 'voc_1based_inclusive') -> list[DetectionBox]:
    """Parse VOC XML. Conversion is explicit; boxes may extend beyond the image."""
    if coordinates not in ('voc_1based_inclusive', 'zero_half_open'):
        raise ValueError('Unsupported XML coordinate convention')
    try:
        root = ET.parse(path).getroot()
        size = root.find('size')
        if size is not None and (int(size.findtext('height', '0')), int(size.findtext('width', '0'))) != image_shape:
            raise ValueError('XML image dimensions do not match input')
        boxes = []
        for obj in root.findall('object'):
            bounds = obj.find('bndbox')
            if bounds is None:
                raise ValueError('XML object missing bndbox')
            values = [float(bounds.findtext(key, 'nan')) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            if any(not math.isfinite(v) or not v.is_integer() for v in values):
                raise ValueError('XML coordinates must be finite integers')
            x0, y0, x1, y1 = map(int, values)
            if coordinates == 'voc_1based_inclusive':
                x0 -= 1
                y0 -= 1
            box = DetectionBox(x0, y0, x1, y1, obj.findtext('name', 'defect'))
            box.validate()
            boxes.append(box)
        return boxes
    except (ET.ParseError, TypeError, OverflowError) as exc:
        raise ValueError('Invalid detector XML') from exc


def binary_cut(foreground_cost: np.ndarray, background_cost: np.ndarray,
               horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    """Minimize D(label) + w*[labels differ], source side = foreground.

    Integer capacities use scale 1024 (rounding error <= 1/2048 per edge).
    Terminal direction is source->pixel D_bg, pixel->sink D_fg, NOT reversed.
    """
    fg, bg = np.asarray(foreground_cost, float), np.asarray(background_cost, float)
    if fg.ndim != 2 or not fg.size or bg.shape != fg.shape:
        raise ValueError('Unary costs need matching nonempty two-dimensional shapes')
    h, w = fg.shape
    horizontal, vertical = np.asarray(horizontal, float), np.asarray(vertical, float)
    if horizontal.shape != (h, w-1) or vertical.shape != (h-1, w):
        raise ValueError('Pairwise costs have incorrect grid shapes')
    arrays = (fg, bg, horizontal, vertical)
    if any(not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1_000_000) for a in arrays):
        raise ValueError('Graph costs must be finite, nonnegative and <= 1000000')
    n = h*w
    ids = np.arange(n).reshape(h, w)
    a = np.concatenate((ids[:, :-1].ravel(), ids[:-1].ravel()))
    b = np.concatenate((ids[:, 1:].ravel(), ids[1:].ravel()))
    weights = np.concatenate((horizontal.ravel(), vertical.ravel()))
    rows = np.concatenate((np.full(n, n), ids.ravel(), a, b))
    cols = np.concatenate((ids.ravel(), np.full(n, n+1), b, a))
    data = np.rint(np.concatenate((bg.ravel(), fg.ravel(), weights, weights))*1024).astype(np.int64)
    graph = coo_matrix((data, (rows, cols)), shape=(n+2, n+2)).tocsr()
    graph.eliminate_zeros()
    flow = maximum_flow(graph, n, n+1)
    residual = graph - flow.flow
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    labels = np.zeros(n+2, bool)
    labels[reachable] = True
    return labels[:n].reshape(h, w)


def _bgr(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim not in (2, 3) or not image.size:
        raise ValueError('Images must be nonempty uint8 gray/BGR/BGRA')
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] not in (3, 4):
        raise ValueError('Images must have 1, 3 or 4 channels')
    return image[..., :3]


def _prototype_distance(features, seeds, groups, scale):
    """Nearest component residual prototype preserves diverse defect colors."""
    distance = np.full(features.shape[:2], np.inf, np.float32)
    for group in groups:
        samples = features[seeds & group]
        if samples.size:
            center = np.median(samples, axis=0)
            candidate = np.mean(((features-center)/scale)**2, axis=2)
            np.minimum(distance, candidate, out=distance)
    return distance


def predict_boxes(input_image: np.ndarray, reference: np.ndarray, valid: np.ndarray,
                  score: np.ndarray, boxes: list[DetectionBox], *, confidence: np.ndarray | None = None,
                  config: BoxGraphCutConfig | None = None) -> BoxPrediction:
    """Independent per-box cuts, unioned without clipping masks to detector boxes.

    Only valid reference pixels can be predicted. Invalid pixels remain unknown,
    not normal; full-image post-hoc evaluation must count uncovered GT as FN.
    High-response components inside each box provide sparse hard foreground;
    only reliable low-response pixels outside *all* boxes provide hard background.
    """
    cfg = config or BoxGraphCutConfig()
    cfg.validate()
    image, normal = _bgr(input_image), _bgr(reference)
    shape = image.shape[:2]
    valid, score = np.asarray(valid), np.asarray(score)
    if (normal.shape != image.shape or valid.shape != shape or score.shape != shape
            or valid.dtype != bool or score.dtype.kind not in 'fiu'
            or not np.isfinite(score).all() or (score < 0).any()):
        raise ValueError('Images, bool validity and finite nonnegative scores must have matching shapes')
    if confidence is None:
        reliable = valid.copy()
    else:
        confidence = np.asarray(confidence)
        if confidence.shape != shape or not np.isin(confidence, [0, 128, 255]).all():
            raise ValueError('Confidence must contain input-sized values 0/128/255')
        reliable = valid & (confidence == 255)
    height, width = shape
    detection_union = np.zeros(shape, bool)
    for box in boxes:
        box.validate()
        detection_union[max(0, box.y0):max(0, min(height, box.y1)),
                        max(0, box.x0):max(0, min(width, box.x1))] = True
    prediction = np.zeros(shape, bool)
    fg_all, bg_all = np.zeros(shape, bool), np.zeros(shape, bool)
    records = []
    for index, box in enumerate(boxes):
        record = dict(index=index, label=box.label, box_xyxy=[box.x0, box.y0, box.x1, box.y1],
                      status='pending', warnings=[], foreground_seed_pixels=0, background_seed_pixels=0,
                      predicted_pixels=0)
        records.append(record)
        if box.x0 >= width or box.y0 >= height or box.x1 <= 0 or box.y1 <= 0:
            record['status'] = 'outside_image'
            continue
        dx = max(cfg.min_context, math.ceil((box.x1-box.x0)*cfg.context_fraction))
        dy = max(cfg.min_context, math.ceil((box.y1-box.y0)*cfg.context_fraction))
        x0, y0 = max(0, box.x0-dx), max(0, box.y0-dy)
        x1, y1 = min(width, box.x1+dx), min(height, box.y1+dy)
        record['roi_xyxy'] = [x0, y0, x1, y1]
        if (x1-x0)*(y1-y0) > cfg.max_roi_pixels:
            record['status'] = 'roi_too_large'
            continue
        roi = np.s_[y0:y1, x0:x1]
        known, trusted, response = valid[roi], reliable[roi], score[roi]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        inside = (xx >= box.x0) & (xx < box.x1) & (yy >= box.y0) & (yy < box.y1)
        record['reference_coverage_fraction'] = float(known[inside].mean())
        if not trusted[inside].any():
            record['status'] = 'no_reliable_reference'
            continue
        count, components = cv2.connectedComponents((inside & trusted & (response >= cfg.seed_threshold)).astype(np.uint8))
        groups = sorted(range(1, count), key=lambda k: (-float(response[components == k].max()), k))[:cfg.max_components]
        fg_seeds = np.zeros_like(known)
        for group in groups:
            indices = np.flatnonzero(components == group)
            take = min(cfg.seeds_per_component, max(1, math.ceil(len(indices)*.1)))
            chosen = indices[np.argsort(-response.ravel()[indices], kind='stable')[:take]]
            fg_seeds.ravel()[chosen] = True
        if not fg_seeds.any():
            record['status'] = 'no_foreground_seeds'
            continue
        bg_seeds = trusted & ~detection_union[roi] & (response <= cfg.background_threshold)
        if not bg_seeds.any():
            record['status'] = 'no_background_seeds'
            continue
        # Signed Lab residuals, not an absolute white/color prior.
        lab = cv2.cvtColor(image[roi], cv2.COLOR_BGR2LAB).astype(np.float32)
        features = lab - cv2.cvtColor(normal[roi], cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_samples = features[bg_seeds]
        center = np.median(bg_samples, axis=0)
        scale = np.maximum(1.4826*np.median(np.abs(bg_samples-center), axis=0), cfg.residual_scale_floor)
        fg_distance = _prototype_distance(features, fg_seeds, (components == k for k in groups), scale)
        bg_distance = np.mean(((features-center)/scale)**2, axis=2)
        appearance_logit = np.clip((bg_distance-fg_distance)*.5, -8, 8)
        anomaly_logit = np.clip((response-cfg.anomaly_midpoint)/cfg.anomaly_scale, -8, 8)
        logit = ((1-cfg.appearance_weight)*anomaly_logit + cfg.appearance_weight*appearance_logit
                 - cfg.outside_penalty*(~inside))
        fg_cost, bg_cost = np.logaddexp(0., -logit), np.logaddexp(0., logit)
        horizontal = cfg.smoothness*(.02 + .98*np.exp(-np.sum(np.diff(lab, axis=1)**2, axis=2)/(2*cfg.contrast_scale**2)))
        vertical = cfg.smoothness*(.02 + .98*np.exp(-np.sum(np.diff(lab, axis=0)**2, axis=2)/(2*cfg.contrast_scale**2)))
        horizontal *= known[:, :-1] & known[:, 1:]
        vertical *= known[:-1] & known[1:]
        # Greater than any possible incident pairwise saving + opposite unary.
        hard = float(max(fg_cost.max(), bg_cost.max()) + 4*cfg.smoothness + 2)
        fg_cost[bg_seeds | ~known], bg_cost[bg_seeds | ~known] = hard, 0
        fg_cost[fg_seeds], bg_cost[fg_seeds] = 0, hard
        mask = binary_cut(fg_cost, bg_cost, horizontal, vertical) & known
        prediction[roi] |= mask
        fg_all[roi] |= fg_seeds
        bg_all[roi] |= bg_seeds
        record.update(status='ok', foreground_seed_pixels=int(fg_seeds.sum()),
                      background_seed_pixels=int(bg_seeds.sum()), predicted_pixels=int(mask.sum()),
                      selected_components=len(groups), available_components=count-1)
        if np.any(mask & inside & ((xx == box.x0) | (xx == box.x1-1) | (yy == box.y0) | (yy == box.y1-1))):
            record['warnings'].append('touches_box')
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            record['warnings'].append('touches_roi')
        if np.any(mask & (cv2.dilate((~known).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)):
            record['warnings'].append('touches_unknown_reference')
        if len(groups) < count-1:
            record['warnings'].append('seed_components_truncated')
    ok = sum(r['status'] == 'ok' for r in records)
    status = ('ok' if ok == len(records) else 'partial') if ok else ('no_boxes' if not records else 'abstained')
    return BoxPrediction(prediction, valid.copy(), fg_all, bg_all, records, status, asdict(cfg))