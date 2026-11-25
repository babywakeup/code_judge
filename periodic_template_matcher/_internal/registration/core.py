"""Global affine registration and comparable multi-template image scores.

Every public matrix maps TEMPLATE pixels to TARGET pixels. Scores are
engineering similarities, not probabilities or proof of absolute cell identity.
This module has no file IO, CLI, label evaluation or visualization dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import os

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils import ValidRegion, build_valid_mask, to_gray
from ...utils.types import Diagnostics, Image, Mask, Matrix, Shape, Size
# Retain existing helper imports for callers of the original core module.
from .correlation import _corr, _periods, _translation_peaks
from .opencv import translation_peaks as opencv_translation_peaks
from .scoring import Features, RefinementLevel, ScoreWarp, _ncc, _refine, _score, prepare_refinement_target


@dataclass(frozen=True)
class PreparedImage:
    """One image at working resolution and its original-to-working transform."""

    shape: Shape
    to_work: Matrix
    scale: float
    features: Features | None


@dataclass(frozen=True)
class RegistrationTarget:
    image: PreparedImage
    lattice: list[Diagnostics]
    refinement: tuple[RefinementLevel, ...] | None


def prepare_registration_target(target: Image, valid: Mask | None, cfg: MatcherConfig,
                                scale: float) -> RegistrationTarget:
    destination = _prepare_image(target, valid, cfg, scale)
    features = destination.features
    lattice = _periods(features[0], features[3]) if features is not None else []
    refinement = prepare_refinement_target(features) if features is not None else None
    return RegistrationTarget(destination, lattice, refinement)


def prepare_registration_sources(templates: Sequence[Image], valids: Sequence[Mask | None],
                                 cfg: MatcherConfig, scale: float) -> tuple[PreparedImage, ...]:
    if cv2.getNumThreads() != 2:
        cv2.setNumThreads(2)
    return tuple(_prepare_image(image, valid, cfg, scale) for image, valid in zip(templates, valids))


def _prepare_image(image: Image, valid: Mask | None, cfg: MatcherConfig,
                   scale: float, *, low_frequency: bool = False) -> PreparedImage:
    gray = to_gray(image)
    mask = build_valid_mask(gray, valid, cfg)
    size = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
    if min(size) < 32:
        raise ValueError('Extreme image aspect/size mismatch; use comparable fields of view')
    to_work = np.diag([size[0] / gray.shape[1], size[1] / gray.shape[0], 1.0])
    gray, mask = ValidRegion.resize_for_matching(gray, mask, size)
    features = (_features(gray, mask, low_frequency=low_frequency, cfg=cfg)
                if np.std(gray[mask > 0]) >= cfg.min_gray_std else None)
    if features is not None and features[2].sum() < cfg.min_edge_pixels:
        features = None
    return PreparedImage(gray.shape, to_work, scale, features)


def _features(gray: Image, mask: Mask, *, low_frequency: bool = False,
              cfg: MatcherConfig | None = None) -> Features:
    # Repair excluded overlays before filtering to avoid leaking text/borders.
    if not mask.all():
        gray = cv2.inpaint(gray, (1 - mask) * 255, 3, cv2.INPAINT_TELEA)
    if low_frequency:
        gray = cv2.GaussianBlur(gray, (0, 0), 1.8)
    x = gray.astype(np.float32) / 255.0
    x = cv2.GaussianBlur(x, (0, 0), 0.8)
    mean = cv2.GaussianBlur(x, (0, 0), 9)
    var = np.maximum(cv2.GaussianBlur(x * x, (0, 0), 9) - mean * mean, 0)
    norm = np.clip((x - mean) / np.sqrt(var + 0.0025), -3, 3)
    dx = cv2.Scharr(x, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(x, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(dx, dy)
    cap = max(float(np.percentile(grad[mask > 0], 95)), 1e-5)
    grad = np.minimum(grad / cap, 1).astype(np.float32)
    safe = cv2.erode(mask, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    cfg = cfg or MatcherConfig()
    thresholds = cfg.low_frequency_canny_thresholds if low_frequency else cfg.canny_thresholds
    edges = ((cv2.Canny(gray, *thresholds) > 0) & (safe > 0)).astype(np.uint8)
    return norm.astype(np.float32) * safe, grad * safe, edges, safe


def _centered_warp(shape: Shape, angle: float, scale: float) -> tuple[Matrix, Size]:
    h, w = shape
    A = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), float(angle), float(scale))
    corners = cv2.transform(np.array([[[0., 0.], [w-1., 0.], [w-1., h-1.], [0., h-1.]]]), A)[0]
    low, high = corners.min(axis=0), corners.max(axis=0)
    A[:, 2] -= low
    return np.vstack([A, [0, 0, 1]]), tuple((np.ceil(high-low).astype(int)+1).tolist())


def _transform_difference(a: Matrix, b: Matrix, shape: Shape) -> float:
    h, w = shape
    pts = np.array([[[w*.2, h*.2], [w*.8, h*.2], [w*.5, h*.5], [w*.2, h*.8], [w*.8, h*.8]]], np.float32)
    return float(np.median(np.linalg.norm(cv2.perspectiveTransform(pts, a) - cv2.perspectiveTransform(pts, b), axis=2)))


def register(template: Image, target: Image, cfg: MatcherConfig | None = None,
             template_valid: Mask | None = None, target_valid: Mask | None = None,
             *, working_scale: float | None = None) -> Diagnostics:
    """Return registration diagnostics; a high score is not an identity proof."""
    cfg = cfg or MatcherConfig()
    cfg.validate()
    scale = cfg.matching_scale(template.shape, target.shape) if working_scale is None else working_scale
    source = _prepare_image(template, template_valid, cfg, scale)
    destination = prepare_registration_target(target, target_valid, cfg, scale)
    return _register_prepared(source, destination.image, cfg, destination.lattice, destination.refinement)


def _coarse_candidates(src: Features, dst: Features, cfg: MatcherConfig) -> list[Matrix]:
    """Search angle/scale/translation hypotheses and keep distinct candidates."""
    shape = src[0].shape
    search = opencv_translation_peaks if cfg.backend == 'opencv' else _translation_peaks
    coarse = []
    for angle in cfg.angles:
        for zoom in cfg.scales:
            A, size = _centered_warp(shape, angle, zoom)
            sg = cv2.warpPerspective(src[1], A, size)
            sm = cv2.warpPerspective(src[3], A, size, flags=cv2.INTER_NEAREST)
            for merit, dx, dy in search(sg, dst[1], sm, dst[3], cfg):
                H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float) @ A
                coarse.append((merit, H))
    shortlist = []
    for _, H in sorted(coarse, key=lambda item: item[0], reverse=True):
        if all(_transform_difference(H, previous, shape) > cfg.coarse_min_separation_px for previous in shortlist):
            shortlist.append(H)
        if len(shortlist) >= cfg.refine_candidates:
            break
    return shortlist


def _refine_candidates(source: PreparedImage, target: PreparedImage,
                       cfg: MatcherConfig, prepared: tuple[RefinementLevel, ...]) -> list[Diagnostics]:
    """Refine, score and deduplicate candidates; retain original-coordinate matrices."""
    src, dst = source.features, target.features
    assert src is not None and dst is not None
    candidates = []
    to_original = np.linalg.inv(target.to_work)
    size = (target.shape[1], target.shape[0])
    for H in _coarse_candidates(src, dst, cfg):
        warp = ScoreWarp(src, H, size)
        metrics = _score(src, dst, H, cfg, warped=warp)
        refined, ecc_ok = _refine(src, dst, H, cfg, prepared=prepared, initial_warp=warp)
        refined_metrics = metrics if np.array_equal(refined, H) else _score(src, dst, refined, cfg)
        if refined_metrics['match_score'] >= metrics['match_score']:
            H, metrics = refined, refined_metrics
        else:
            ecc_ok = False
        if all(_transform_difference(H, np.asarray(c['_H_work']), source.shape) > cfg.refined_min_separation_px
               for c in candidates):
            metrics.update(_H_work=H.tolist(), ecc_converged=bool(ecc_ok),
                           matrix_template_to_target=(to_original @ H @ source.to_work).tolist())
            candidates.append(metrics)
    return candidates


def _register_prepared(source: PreparedImage, target: PreparedImage,
                       cfg: MatcherConfig, lattice: list[Diagnostics],
                       refinement: tuple[RefinementLevel, ...] | None = None) -> Diagnostics:
    """Match prepared images without repeating boundary validation or preprocessing."""
    size = [target.shape[1], target.shape[0]]
    fail = dict(match_score=0.0, geometry_accepted=False, absolute_cell_verified=False,
                periodic_detected=False, matrix_template_to_target=None, candidates=[],
                working_scale=target.scale, working_target_size=size,
                reasons=['insufficient_texture_or_overlap'])
    if source.features is None or target.features is None:
        return fail
    if refinement is None:
        refinement = prepare_refinement_target(target.features)
    candidates = _refine_candidates(source, target, cfg, refinement)
    if not candidates:
        return fail
    candidates.sort(key=lambda c: c['match_score'], reverse=True)
    best = candidates[0].copy()
    gap = best['match_score'] - candidates[1]['match_score'] if len(candidates) > 1 else None
    # Autocorrelation plus far-apart near-equal matches flag periodic ambiguity.
    alias = any(best['match_score'] - c['match_score'] < cfg.min_candidate_gap and
                _transform_difference(np.asarray(best['_H_work']), np.asarray(c['_H_work']), source.shape) > min(source.shape)*0.12
                for c in candidates[1:])
    periodic = bool(lattice or alias)
    accepted = best['match_score'] >= cfg.min_score and best['target_coverage'] >= cfg.min_overlap
    best.pop('_H_work', None)
    for c in candidates:
        c.pop('_H_work', None)
    best.update(geometry_accepted=bool(accepted), periodic_detected=periodic,
                absolute_cell_verified=bool(accepted and not periodic and gap is not None and gap >= cfg.min_candidate_gap),
                absolute_identity_status='unverified_periodic' if periodic else 'visual_evidence_only',
                candidate_score_gap=gap, periodic_alias_detected=bool(alias), lattice_candidates=lattice,
                working_scale=target.scale, working_target_size=size, candidates=candidates,
                reasons=[] if accepted else ['score_or_coverage_below_threshold'])
    if periodic:
        best['reasons'].append('absolute_period_index_requires_external_anchor')
    return best


def _template_worker_count(template_count: int, requested: int) -> int:
    """Bound automatic parallelism and avoid workers without a template to match."""
    return max(1, min(template_count, os.cpu_count() or 1, requested or 4))


def _verified_score(result):
    return result['geometry_accepted'], result['match_score']


def _score_gap(ranked):
    return ranked[0]['match_score'] - ranked[1]['match_score'] if len(ranked) > 1 else None


def _rescore_low_frequency(matches, templates, target, template_valids, target_valid, cfg, scale):
    """Verify the existing candidate poses against coarser image structure."""
    dst = _prepare_image(target, target_valid, cfg, scale, low_frequency=True)
    if dst.features is None:
        return None
    ranked = []
    for match, image, valid in zip(matches, templates, template_valids):
        rescored = dict(match, geometry_accepted=False, match_score=0.)
        ranked.append(rescored)
        if not match['candidates']:
            continue
        src = _prepare_image(image, valid, cfg, scale, low_frequency=True)
        if src.features is None:
            continue
        candidates = []
        for candidate in match['candidates']:
            work = dst.to_work @ np.asarray(candidate['matrix_template_to_target']) @ np.linalg.inv(src.to_work)
            metrics = _score(src.features, dst.features, work, cfg)
            accepted = (metrics['match_score'] >= cfg.min_score and metrics['target_coverage'] >= cfg.min_overlap
                        and metrics['structure_ncc'] >= cfg.robust_min_ncc
                        and metrics['local_consistency'] >= cfg.robust_min_local_consistency)
            candidates.append(dict(candidate, **metrics, geometry_accepted=bool(accepted),
                                   original_match_score=candidate['match_score']))
        candidates.sort(key=_verified_score, reverse=True)
        rescored.update(candidates[0], candidates=candidates,
                        candidate_score_gap=_score_gap([c for c in candidates if c['geometry_accepted']]),
                        evidence_mode='low_frequency', absolute_cell_verified=False)
        if rescored['geometry_accepted']:
            rescored['reasons'] = [reason for reason in match['reasons']
                                   if reason != 'score_or_coverage_below_threshold']
    return ranked


def select_templates(templates: Sequence[Image], target: Image,
                     cfg: MatcherConfig | None = None,
                     template_valids: Sequence[Mask | None] | None = None,
                     target_valid: Mask | None = None,
                     label_mode: str = 'semantic', *,
                     _prepared_sources: Sequence[PreparedImage] | None = None,
                     _prepared_target: RegistrationTarget | None = None) -> Diagnostics:
    """Rank templates on the same target ROI, scale and score definition.

    semantic allows shared cell classes; instance requires verified identity.
    Templates run independently; map preserves input order and stable score ties.
    """
    cfg = cfg or MatcherConfig()
    cfg.validate()
    if not templates or label_mode not in ('semantic', 'instance'):
        raise ValueError('Provide templates and label_mode semantic/instance')
    if template_valids is None:
        template_valids = [None]*len(templates)
    if len(template_valids) != len(templates):
        raise ValueError('One valid mask per template is required')
    cv2.setRNGSeed(cfg.seed)
    if cv2.getNumThreads() != 2:
        cv2.setNumThreads(2)
    common_scale = cfg.matching_scale(*(im.shape for im in [*templates, target]))
    prepared = _prepared_target or prepare_registration_target(target, target_valid, cfg, common_scale)
    lattice = deepcopy(prepared.lattice) if _prepared_target is not None else prepared.lattice

    def match_one(i: int) -> Diagnostics:
        template, valid = templates[i], template_valids[i]
        source = (_prepare_image(template, valid, cfg, common_scale)
                  if _prepared_sources is None else _prepared_sources[i])
        result = _register_prepared(source, prepared.image, cfg, lattice, prepared.refinement)
        result['template_index'] = i
        return result

    workers = _template_worker_count(len(templates), cfg.template_workers)
    if workers == 1:
        matches = [match_one(i) for i in range(len(templates))]
    else:
        # Target features/lattice are read-only; every task owns its source and
        # candidate arrays. Keep OpenCV runtime setup above, outside the pool.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='template-match') as pool:
            matches = list(pool.map(match_one, range(len(templates))))
    ranked = sorted(matches, key=lambda r: r['match_score'], reverse=True)
    gap = _score_gap(ranked)
    if (cfg.robust_appearance and label_mode == 'semantic' and not ranked[0]['geometry_accepted']
            and any(match['candidates'] for match in matches)):
        fallback = _rescore_low_frequency(matches, templates, target, template_valids, target_valid, cfg, common_scale)
        if fallback is not None and any(match['geometry_accepted'] for match in fallback):
            matches = fallback
            ranked = sorted(matches, key=_verified_score, reverse=True)
            gap = _score_gap([match for match in ranked if match['geometry_accepted']])
    best = ranked[0]
    chosen = best['template_index'] if best['geometry_accepted'] else None
    separated = gap is None or gap >= cfg.min_template_gap
    visual_ok = bool(chosen is not None and separated)
    absolute_ok = bool(visual_ok and best['absolute_cell_verified'])
    accepted = absolute_ok if label_mode == 'instance' else visual_ok
    status = 'accepted' if accepted else ('low_match_score' if chosen is None else
             'templates_too_close' if not separated else 'absolute_cell_unverified')
    return dict(best_template_index=chosen, accepted=accepted, status=status,
                evidence_mode=best.get('evidence_mode', 'standard'),
                label_mode=label_mode, template_score_gap=gap,
                absolute_mapping_accepted=absolute_ok, ranking=[r['template_index'] for r in ranked],
                matches=matches, config=asdict(cfg),
                score_definition='100*(0.55*structure_ncc+0.30*edge_alignment+0.15*local_consistency)*sqrt(target_coverage)',
                score_is_probability=False)
