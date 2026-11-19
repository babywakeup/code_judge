"""Color, structure and multiscale evidence; scores are not probabilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from periodic_template_matcher.utils import to_gray, validate_image
from .salience import compute_salience
from .ssim import multiscale_ssim


@dataclass(frozen=True)
class AnomalyConfig:
    tolerance_radius: int = 2
    seam_radius: int = 3
    high_threshold: float = 1.8
    low_threshold: float = 1.
    structure_weight: float = 1.5

    def validate(self) -> None:
        for name, limit in (('tolerance_radius', 5), ('seam_radius', 32)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= limit:
                raise ValueError(f'{name} must be an integer in [0, {limit}]')
        if (not np.isfinite([self.high_threshold, self.low_threshold]).all()
                or not 0 < self.low_threshold <= self.high_threshold):
            raise ValueError('Thresholds must be finite and 0 < low <= high')
        if not np.isfinite(self.structure_weight) or self.structure_weight <= 0:
            raise ValueError('structure_weight must be finite and positive')


@dataclass
class AnomalyResult:
    strict_score: np.ndarray
    tolerant_score: np.ndarray
    color_response: np.ndarray
    structure_response: np.ndarray
    multiscale_response: np.ndarray
    comparison_valid: np.ndarray
    confidence: np.ndarray
    candidate_mask: np.ndarray
    priority_mask: np.ndarray
    metadata: dict
    salience_score: np.ndarray
    salience_weight: np.ndarray
    salience_mask: np.ndarray
    ssim_response: np.ndarray
    ssim_score: np.ndarray
    ssim_valid: np.ndarray
    ssim_windows: dict
    normal_score: np.ndarray
    normal_valid: np.ndarray
    supported_salience_score: np.ndarray


def _mask(array: np.ndarray, shape: tuple, name: str) -> np.ndarray:
    if (not isinstance(array, np.ndarray) or array.shape != shape
            or array.dtype.kind not in 'buif' or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be a finite numeric/bool mask matching image dimensions')
    return array > 0


def _features(image: np.ndarray, valid: np.ndarray) -> tuple:
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image[..., :3]
    lab = cv2.cvtColor(bgr.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
    lab[..., 0] *= 2.55  # L in intensity-like units; a/b retain Lab units.
    weight = cv2.GaussianBlur(valid.astype(np.float32), (7, 7), 1.2)
    smooth = cv2.GaussianBlur(lab*valid[..., None], (7, 7), 1.2)/np.maximum(weight[..., None], 1e-6)
    gray = to_gray(image).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)/8
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)/8
    return lab, smooth, gx, gy


def _responses(a: tuple, b: tuple, gradient_valid: np.ndarray) -> tuple:
    color = np.linalg.norm(a[0]-b[0], axis=2)
    smooth = np.linalg.norm(a[1]-b[1], axis=2)
    structure = np.hypot(a[2]-b[2], a[3]-b[3])
    structure[~gradient_valid] = 0
    return color, structure, smooth


def _calibration(response: np.ndarray, valid: np.ndarray, floor: float) -> dict:
    values = response[valid]
    if len(values) > 100000:
        values = values[::max(1, len(values)//100000)]
    median = float(np.median(values)) if len(values) else 0.
    sigma = float(1.4826*np.median(np.abs(values-median))) if len(values) else 0.
    # Do not subtract the median: a broad consistent change is still evidence.
    # Bound adaptation so a heavily anomalous input cannot normalize itself away.
    scale = float(np.clip(4.5*sigma, floor, 3*floor))
    return dict(scale=scale, median=median, robust_sigma=sigma, samples=len(values), floor=floor)


def _combine(responses: tuple, scales: list) -> np.ndarray:
    score = responses[0]/scales[0]
    for response, scale in zip(responses[1:], scales[1:]):
        np.maximum(score, response/scale, out=score)
    return score


def _hysteresis(score: np.ndarray, valid: np.ndarray, cfg: AnomalyConfig) -> np.ndarray:
    support = valid & (score >= cfg.low_threshold)
    seeds = valid & (score >= cfg.high_threshold)
    count, labels = cv2.connectedComponents(support.astype(np.uint8))
    keep = np.zeros(count, bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    return keep[labels]


def analyze_pair(input_image: np.ndarray, reference: np.ndarray, reference_valid: np.ndarray, *,
                 photometric_valid: np.ndarray | None = None, source_map: np.ndarray | None = None,
                 config: AnomalyConfig | None = None, normal_baseline=None,
                 semantic_labels: np.ndarray | None = None,
                 semantic_valid: np.ndarray | None = None) -> AnomalyResult:
    """Compare aligned images. Unknown pixels have zero scores AND false validity.

    Strict evidence remains intact when a small displacement explains a signal.
    The tolerant branch minimizes a joint score at ONE common source displacement.
    """
    cfg = config if config is not None else AnomalyConfig()
    cfg.validate()
    validate_image(input_image)
    validate_image(reference)
    shape = input_image.shape[:2]
    if reference.shape[:2] != shape:
        raise ValueError('Input and reference must have the same image dimensions')
    valid = _mask(reference_valid, shape, 'reference_valid')
    if photometric_valid is not None:
        valid &= _mask(photometric_valid, shape, 'photometric_valid')
    if (input_image.ndim == 3 and input_image.shape[2] == 4):
        valid &= input_image[..., 3] > 0
    if (reference.ndim == 3 and reference.shape[2] == 4):
        valid &= reference[..., 3] > 0
    if source_map is not None:
        if (source_map.shape != shape or source_map.dtype.kind not in 'ui'
                or (source_map < 0).any() or np.any(valid & (source_map == 0))):
            raise ValueError('source_map must contain positive integer source IDs at valid pixels')
        ids = source_map
    else:
        ids = valid.astype(np.uint16)
    seam = np.zeros(shape, bool)
    different = ids[:, 1:] != ids[:, :-1]
    seam[:, 1:] |= different
    seam[:, :-1] |= different
    different = ids[1:] != ids[:-1]
    seam[1:] |= different
    seam[:-1] |= different
    seam |= ~valid
    if cfg.seam_radius:
        size = 2*cfg.seam_radius+1
        seam = cv2.dilate(seam.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
    confidence = np.where(valid, np.where(seam, 128, 255), 0).astype(np.uint8)
    calibration_valid = valid & ~seam
    if calibration_valid.sum() < 64:
        calibration_valid = valid
    gradient_valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    a, b = _features(input_image, valid), _features(reference, valid)
    responses = _responses(a, b, gradient_valid)
    for response in responses:
        response[~valid] = 0
    names = ('color', 'structure', 'multiscale')
    calibration = {name: _calibration(response, calibration_valid, floor)
                   for name, response, floor in zip(names, responses, (10., 6., 8.))}
    scales = [calibration[name]['scale'] for name in names]
    scales[1] /= cfg.structure_weight
    strict = _combine(responses, scales)
    tolerant = strict.copy()
    h, w = shape
    for dy in range(-cfg.tolerance_radius, cfg.tolerance_radius+1):
        for dx in range(-cfg.tolerance_radius, cfg.tolerance_radius+1):
            if dx == dy == 0:
                continue
            y0, y1 = max(0, -dy), min(h, h-dy)
            x0, x1 = max(0, -dx), min(w, w-dx)
            dst = np.s_[y0:y1, x0:x1]
            src = np.s_[y0+dy:y1+dy, x0+dx:x1+dx]
            allowed = valid[dst] & valid[src] & (ids[dst] == ids[src])
            # A missing gradient neighborhood must not falsely improve a score.
            allowed &= gradient_valid[dst] & gradient_valid[src]
            shifted = _responses(tuple(f[dst] for f in a), tuple(f[src] for f in b),
                                 gradient_valid[dst] & gradient_valid[src])
            score = _combine(shifted, scales)
            view = tolerant[dst]
            np.minimum(view, np.where(allowed, score, np.inf), out=view)
    strict[~valid] = tolerant[~valid] = 0
    candidates = _hysteresis(strict, valid, cfg)
    priority = _hysteresis(tolerant, valid & (confidence == 255), cfg) & candidates
    texture = np.hypot(b[2], b[3])
    texture[~gradient_valid] = 0
    texture = cv2.GaussianBlur(texture, (5, 5), 1.)
    salience, salience_weight, salience_info = compute_salience(strict, tolerant, texture, valid, confidence)
    salience_mask = _hysteresis(salience, valid, AnomalyConfig(high_threshold=3., low_threshold=1.5)) & candidates
    ssim_response, ssim_valid, ssim_windows = multiscale_ssim(input_image, reference, valid, ids)
    ssim_score = (ssim_response/.15*salience_weight).astype(np.float32)
    normal_score, normal_valid = np.zeros(shape, np.float32), np.zeros(shape, bool)
    normal_info = dict(status='not_requested')
    if normal_baseline is not None:
        normal_comparable = valid & ssim_valid
        if semantic_valid is not None:
            normal_comparable &= _mask(semantic_valid, shape, 'semantic_valid')
        normal_score, normal_valid, normal_info = normal_baseline.score(
            (*responses, ssim_response), texture, normal_comparable, semantic_labels,
            salience_weight, cfg.structure_weight)
    support = np.clip(ssim_score/3., 0., 1.)
    support[~ssim_valid] = 0
    np.maximum(support, np.where(normal_valid, np.clip(normal_score/3., 0., 1.), 0.), out=support)
    supported_salience = (salience*(1+support)).astype(np.float32)
    metadata = dict(format_version=1, config=asdict(cfg), image_shape=list(shape),
                    score_meaning='relative_evidence_not_probability',
                    status='available' if valid.any() else 'no_comparable_pixels',
                    valid_pixels=int(valid.sum()), high_confidence_pixels=int((confidence == 255).sum()),
                    candidate_pixels=int(candidates.sum()), priority_pixels=int(priority.sum()),
                    calibration=calibration,
                    salience=salience_info, salience_pixels=int(salience_mask.sum()),
                    ssim=dict(method='multi_window_full_ssim', windows=[5, 11, 21],
                              response_scale=.15, valid_pixels=int(ssim_valid.sum())),
                    normal_calibration=normal_info,
                    supported_salience=dict(method='bounded_structural_support', maximum_multiplier=2.,
                                            note='Support can amplify existing salience only; zero salience remains zero.'),
                    limitations=['Residual registration and normal texture can produce false candidates.',
                                 'Changes absorbed by upstream photometry cannot be recovered from this pair alone.'])
    return AnomalyResult(strict, tolerant, *responses, valid, confidence, candidates, priority, metadata,
                         salience, salience_weight, salience_mask,
                         ssim_response, ssim_score, ssim_valid, ssim_windows, normal_score, normal_valid,
                         supported_salience)
