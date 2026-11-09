"""Shared geometric verification: structural scores and guarded ECC refinement."""

import math

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils.types import Array, Diagnostics, Matrix


Features = tuple[Array, Array, Array, Array]  # Normalized intensity, gradient, edges, ROI.


def _ncc(x: Array, y: Array) -> float:
    if len(x) < 32:
        return 0.0
    x, y = x - x.mean(), y - y.mean()
    return float(np.clip(np.dot(x, y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-9), 0, 1))


def _score(src: Features, dst: Features, H: Matrix, cfg: MatcherConfig) -> Diagnostics:
    sn, _, se, sm = src
    tn, _, te, tm = dst
    size = (tn.shape[1], tn.shape[0])
    valid = (cv2.warpPerspective(sm, H, size, flags=cv2.INTER_NEAREST) > 0) & (tm > 0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    coverage = float(valid.sum() / max(tm.sum(), 1))
    empty = dict(match_score=0.0, structure_ncc=0.0, edge_alignment=0.0,
                 local_consistency=0.0, target_coverage=coverage, edge_distance_p95_work_px=None)
    if coverage < cfg.min_overlap:
        return empty
    wn = cv2.warpPerspective(sn, H, size)
    ncc = _ncc(wn[valid], tn[valid])
    we = (cv2.warpPerspective(se, H, size, flags=cv2.INTER_NEAREST) > 0) & valid
    et = (te > 0) & valid
    if we.sum() < 40 or et.sum() < 40:
        return empty
    dt = cv2.distanceTransform((~et).astype(np.uint8), cv2.DIST_L2, 5)
    ds = cv2.distanceTransform((~we).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([dt[we], ds[et]])
    # Symmetric: matching a tiny subset of edges cannot earn a perfect score.
    edge = float(np.mean(np.exp(-distances / cfg.edge_tolerance_px)))
    tile_scores = []
    for y0, y1 in zip(np.linspace(0, tn.shape[0], 5, dtype=int)[:-1], np.linspace(0, tn.shape[0], 5, dtype=int)[1:]):
        for x0, x1 in zip(np.linspace(0, tn.shape[1], 5, dtype=int)[:-1], np.linspace(0, tn.shape[1], 5, dtype=int)[1:]):
            v = valid[y0:y1, x0:x1]
            if v.sum() >= max(100, v.size * 0.4) and np.std(tn[y0:y1, x0:x1][v]) > 0.05:
                tile_scores.append(_ncc(wn[y0:y1, x0:x1][v], tn[y0:y1, x0:x1][v]))
    local = float(np.percentile(tile_scores, 25)) if tile_scores else 0.0
    total = 100 * (0.55*ncc + 0.30*edge + 0.15*local) * math.sqrt(coverage)
    return dict(match_score=float(np.clip(total, 0, 100)), structure_ncc=ncc,
                edge_alignment=edge, local_consistency=local, target_coverage=coverage,
                edge_distance_p95_work_px=float(np.percentile(distances, 95)))


def _refine(src: Features, dst: Features, H: Matrix, cfg: MatcherConfig) -> tuple[Matrix, bool]:
    sn, _, _, sm = src
    tn, _, _, tm = dst
    current = H.copy()
    converged = False
    for factor in (0.5, 1.0):
        size = (max(32, round(tn.shape[1]*factor)), max(32, round(tn.shape[0]*factor)))
        D = np.diag([size[0]/tn.shape[1], size[1]/tn.shape[0], 1.0])
        hs = D @ current
        warped = cv2.warpPerspective(sn, hs, size)
        valid = cv2.warpPerspective(sm, hs, size, flags=cv2.INTER_NEAREST)
        valid &= cv2.resize(tm, size, interpolation=cv2.INTER_NEAREST)
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
        if valid.sum() < 1000:
            continue
        target = cv2.resize(tn, size, interpolation=cv2.INTER_AREA)
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(target, warped, W, cv2.MOTION_AFFINE,
                                      (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                                       cfg.ecc_iterations, 1e-5), valid*255, 5)
            # ECC maps target -> already-warped source coordinates.
            updated = np.linalg.inv(D) @ np.linalg.inv(np.vstack([W, [0, 0, 1]])) @ D @ current
            singular = np.linalg.svd(updated[:2, :2], compute_uv=False)
            if np.linalg.det(updated[:2, :2]) > 0 and singular.min() > 0.65 and singular.max() < 1.5:
                current, converged = updated, True
        except (cv2.error, np.linalg.LinAlgError):
            # Preserve the coarse candidate; never report success on exception.
            pass
    return current, converged

