"""Shared geometric verification: structural scores and guarded ECC refinement."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils.types import Array, Diagnostics, Matrix


Features = tuple[Array, Array, Array, Array]  # Normalized intensity, gradient, edges, ROI.


class ScoreWarp:
    """Lazy, candidate-local projections reused across independent region scores.

    Only source projections are cached. Target validity, symmetric edge distance
    and NCC are recomputed for each region, so evidence cannot leak across ROIs.
    No state is shared between images, transforms, or matcher calls.
    """

    def __init__(self, src: Features, H: Matrix, size: tuple[int, int], *,
                 intensity: Array | None = None, support: Array | None = None):
        self.src, self.H, self.size = src, H, size
        self._intensity, self._support, self._edges = intensity, support, None

    @property
    def support(self) -> Array:
        if self._support is None:
            self._support = cv2.warpPerspective(self.src[3], self.H, self.size, flags=cv2.INTER_NEAREST)
        return self._support

    @property
    def intensity(self) -> Array:
        if self._intensity is None:
            self._intensity = cv2.warpPerspective(self.src[0], self.H, self.size)
        return self._intensity

    @property
    def edges(self) -> Array:
        if self._edges is None:
            self._edges = cv2.warpPerspective(self.src[2], self.H, self.size, flags=cv2.INTER_NEAREST)
        return self._edges


def _ncc(x: Array, y: Array) -> float:
    if len(x) < 32:
        return 0.0
    x, y = x - x.mean(), y - y.mean()
    return float(np.clip(np.dot(x, y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-9), 0, 1))


def _edge_distances(source_edges: Array, target_edges: Array) -> Array:
    """Keep the full canvas so the OpenCV backend preserves legacy rounding."""
    dt = cv2.distanceTransform((~target_edges).astype(np.uint8), cv2.DIST_L2, 5)
    ds = cv2.distanceTransform((~source_edges).astype(np.uint8), cv2.DIST_L2, 5)
    return np.concatenate([dt[source_edges], ds[target_edges]])


def _score(src: Features, dst: Features, H: Matrix, cfg: MatcherConfig, *,
           warped: ScoreWarp | None = None, min_structure_ncc: float | None = None) -> Diagnostics:
    tn, _, te, tm = dst
    size = (tn.shape[1], tn.shape[0])
    warped = warped if warped is not None else ScoreWarp(src, H, size)
    valid = (warped.support > 0) & (tm > 0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    coverage = float(valid.sum() / max(tm.sum(), 1))
    empty = dict(match_score=0.0, structure_ncc=0.0, edge_alignment=0.0,
                 local_consistency=0.0, target_coverage=coverage, edge_distance_p95_work_px=None)
    if coverage < cfg.min_overlap:
        return empty
    wn = warped.intensity
    ncc = _ncc(wn[valid], tn[valid])
    if min_structure_ncc is not None and ncc < min_structure_ncc:
        # Local acceptance requires both gates. An NCC failure cannot be
        # rescued by the more expensive symmetric edge-distance calculation.
        return dict(empty, structure_ncc=ncc, edge_evaluated=False)
    we = (warped.edges > 0) & valid
    et = (te > 0) & valid
    if we.sum() < cfg.min_edge_pixels or et.sum() < cfg.min_edge_pixels:
        return empty
    distances = _edge_distances(we, et)
    # Symmetric: matching a tiny subset of edges cannot earn a perfect score.
    edge = float(np.mean(np.exp(-distances / cfg.edge_tolerance_px)))
    tile_scores = []
    ys = np.linspace(0, tn.shape[0], 5, dtype=int)
    xs = np.linspace(0, tn.shape[1], 5, dtype=int)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        for x0, x1 in zip(xs[:-1], xs[1:]):
            v = valid[y0:y1, x0:x1]
            if v.sum() >= max(100, v.size * 0.4) and np.std(tn[y0:y1, x0:x1][v]) > 0.05:
                tile_scores.append(_ncc(wn[y0:y1, x0:x1][v], tn[y0:y1, x0:x1][v]))
    local = float(np.percentile(tile_scores, 25)) if tile_scores else 0.0
    total = 100 * (0.55*ncc + 0.30*edge + 0.15*local) * math.sqrt(coverage)
    return dict(match_score=float(np.clip(total, 0, 100)), structure_ncc=ncc,
                edge_alignment=edge, local_consistency=local, target_coverage=coverage,
                edge_distance_p95_work_px=float(np.percentile(distances, 95)))


@dataclass(frozen=True)
class RefinementLevel:
    size: tuple[int, int]
    to_level: Matrix
    intensity: Array
    support: Array


def prepare_refinement_target(dst: Features) -> tuple[RefinementLevel, ...]:
    """Build the target-only ECC pyramid once for candidates sharing its ROI."""
    tn, _, _, tm = dst
    levels = []
    for factor in (0.5, 1.0):
        size = (max(32, round(tn.shape[1]*factor)), max(32, round(tn.shape[0]*factor)))
        D = np.diag([size[0]/tn.shape[1], size[1]/tn.shape[0], 1.0])
        intensity = cv2.resize(tn, size, interpolation=cv2.INTER_AREA)
        support = cv2.resize(tm, size, interpolation=cv2.INTER_NEAREST)
        levels.append(RefinementLevel(size, D, intensity, support))
    return tuple(levels)


def refinement_for_support(levels: tuple[RefinementLevel, ...], support: Array) -> tuple[RefinementLevel, ...]:
    """Share target intensities, never another occurrence's local ECC mask."""
    return tuple(RefinementLevel(level.size, level.to_level, level.intensity,
                                cv2.resize(support, level.size, interpolation=cv2.INTER_NEAREST))
                 for level in levels)


def _ecc_affine(target: Array, source: Array, valid: Array, cfg: MatcherConfig) -> Matrix:
    return _ecc_transform(target, source, valid, cfg, cv2.MOTION_AFFINE)


def _ecc_translation(target: Array, source: Array, valid: Array, cfg: MatcherConfig) -> Matrix:
    return _ecc_transform(target, source, valid, cfg, cv2.MOTION_TRANSLATION)


def _ecc_transform(target: Array, source: Array, valid: Array, cfg: MatcherConfig, motion: int) -> Matrix:
    """Optionally solve sparse-support ECC in original-resolution coordinates.

    Cropping changes floating-point reduction order and can change convergence.
    It is explicitly opt-in. Failure or motion escaping the padded window falls
    back to the unchanged full-frame call; callers still apply their original
    full-canvas score, geometry and displacement acceptance rules.
    """
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, cfg.ecc_iterations, cfg.ecc_epsilon)

    def solve(t, s, mask):
        _, warp = cv2.findTransformECC(t, s, np.eye(2, 3, dtype=np.float32),
                                      motion, criteria, mask * 255, 5)
        return warp

    if cfg.ecc_roi:
        x, y, width, height = cv2.boundingRect(valid)
        h, w = valid.shape
        margin = 24  # Work/level pixels: filtering halo plus local motion room.
        x0, y0 = max(0, x-margin), max(0, y-margin)
        x1, y1 = min(w, x+width+margin), min(h, y+height+margin)
        if min(x1-x0, y1-y0) >= 32 and (x1-x0)*(y1-y0) < .7*w*h:
            roi = np.s_[y0:y1, x0:x1]
            try:
                # Copies prevent OpenCV treating a submatrix's parent as its
                # filtering border; all artificial borders have a 24 px halo.
                local = solve(target[roi].copy(), source[roi].copy(), valid[roi].copy())
                T = np.array([[1., 0., x0], [0., 1., y0], [0., 0., 1.]])
                full = T @ np.vstack([local, [0., 0., 1.]]) @ np.linalg.inv(T)
                corners = np.array([[[x, y], [x+width-1, y],
                                      [x+width-1, y+height-1], [x, y+height-1]]], np.float32)
                mapped = cv2.perspectiveTransform(corners, full)[0]
                # Only artificial borders need a halo. Real image boundaries
                # retain the same zero/border handling as full-frame ECC.
                safe = (np.isfinite(full).all() and np.isfinite(mapped).all()
                        and (x0 == 0 or mapped[:, 0].min() >= x0+4)
                        and (x1 == w or mapped[:, 0].max() < x1-4)
                        and (y0 == 0 or mapped[:, 1].min() >= y0+4)
                        and (y1 == h or mapped[:, 1].max() < y1-4))
                if safe:
                    return full[:2]
            except (cv2.error, np.linalg.LinAlgError):
                pass
    return solve(target, source, valid)


def _refine(src: Features, dst: Features, H: Matrix, cfg: MatcherConfig, *,
            prepared: tuple[RefinementLevel, ...] | None = None,
            initial_warp: ScoreWarp | None = None,
            motion: int = cv2.MOTION_AFFINE) -> tuple[Matrix, bool]:
    sn, _, _, sm = src
    current = H.copy()
    converged = False
    if prepared is None:
        prepared = prepare_refinement_target(dst)
    for level in prepared:
        size, D = level.size, level.to_level
        hs = D @ current
        reuse = initial_warp is not None and size == initial_warp.size and np.array_equal(hs, initial_warp.H)
        projected_support = (initial_warp.support if reuse else
                             cv2.warpPerspective(sm, hs, size, flags=cv2.INTER_NEAREST))
        valid = projected_support & level.support
        valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
        if valid.sum() < 1000:
            continue
        warped = initial_warp.intensity if reuse else cv2.warpPerspective(sn, hs, size)
        try:
            solver = _ecc_affine if motion == cv2.MOTION_AFFINE else _ecc_translation
            W = solver(level.intensity, warped, valid, cfg)
            # ECC maps target -> already-warped source coordinates.
            updated = np.linalg.inv(D) @ np.linalg.inv(np.vstack([W, [0, 0, 1]])) @ D @ current
            singular = np.linalg.svd(updated[:2, :2], compute_uv=False)
            if np.linalg.det(updated[:2, :2]) > 0 and singular.min() > 0.65 and singular.max() < 1.5:
                current, converged = updated, True
        except (cv2.error, np.linalg.LinAlgError):
            # Preserve the coarse candidate; never report success on exception.
            pass
    return current, converged

