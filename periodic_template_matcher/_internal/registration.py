"""Global affine registration and comparable multi-template image scores.

Every public matrix maps TEMPLATE pixels to TARGET pixels. Scores are
engineering similarities, not probabilities or proof of absolute cell identity.
This module has no file IO, CLI, label evaluation or visualization dependencies.
"""
from dataclasses import asdict
import math

import cv2
import numpy as np

from .opencv_registration import translation_peaks as opencv_translation_peaks
from ..config import MatcherConfig
from ..utils import ValidRegion, build_valid_mask, to_gray


def _features(gray, mask):
    x = gray.astype(np.float32) / 255.0
    # Repair excluded overlays before filtering to avoid leaking text/borders.
    if not mask.all():
        repaired = cv2.inpaint(gray, (1 - mask) * 255, 3, cv2.INPAINT_TELEA)
        gray = repaired
        x = repaired.astype(np.float32) / 255.0
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
    edges = ((cv2.Canny(gray, 35, 100) > 0) & (safe > 0)).astype(np.uint8)
    return norm.astype(np.float32) * safe, grad * safe, edges, safe


def _corr(a, b, fft_shape):
    fa = np.fft.rfft2(a, s=fft_shape)
    fb = np.fft.rfft2(b, s=fft_shape)
    return np.fft.irfft2(fa * np.conj(fb), s=fft_shape).real


def _translation_peaks(source, target, sm, tm, cfg):
    """Linear (not circular) normalized correlation, multiple displacement peaks."""
    sh, sw = source.shape
    th, tw = target.shape
    shape = (cv2.getOptimalDFTSize(sh + th - 1), cv2.getOptimalDFTSize(sw + tw - 1))
    ys, xs = np.arange(-sh + 1, th), np.arange(-sw + 1, tw)
    index = np.ix_(ys % shape[0], xs % shape[1])
    overlap = _corr(tm, sm, shape)[index]
    num = _corr(target, source, shape)[index]
    den1 = _corr(target * target, sm, shape)[index]
    den2 = _corr(tm, source * source, shape)[index]
    corr = num / np.sqrt(np.maximum(den1 * den2, 1e-8))
    cov = np.maximum(overlap, 0) / max(float(tm.sum()), 1)
    eligible = (cov >= cfg.min_overlap) & (overlap >= cfg.min_overlap * sm.sum())
    corr = np.where(eligible, np.clip(corr, 0, 1) * np.sqrt(np.minimum(cov, 1)), -1)
    peaks = []
    radius = max(8, min(sh, sw, th, tw) // 20)
    for _ in range(cfg.peaks_per_hypothesis):
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        if corr[y, x] < 0:
            break
        peaks.append((float(corr[y, x]), float(xs[x]), float(ys[y])))
        corr[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = -1
    return peaks


def _centered_warp(shape, angle, scale):
    h, w = shape
    A = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), float(angle), float(scale))
    corners = cv2.transform(np.array([[[0., 0.], [w-1., 0.], [w-1., h-1.], [0., h-1.]]]), A)[0]
    low, high = corners.min(axis=0), corners.max(axis=0)
    A[:, 2] -= low
    return np.vstack([A, [0, 0, 1]]), tuple((np.ceil(high-low).astype(int)+1).tolist())


def _ncc(x, y):
    if len(x) < 32:
        return 0.0
    x, y = x - x.mean(), y - y.mean()
    return float(np.clip(np.dot(x, y) / max(np.linalg.norm(x)*np.linalg.norm(y), 1e-9), 0, 1))


def _score(src, dst, H, cfg):
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


def _refine(src, dst, H, cfg):
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


def _periods(feature, mask):
    """Diagnostic lattice candidates; harmonics are not unique cell identities."""
    h, w = feature.shape
    shape = (cv2.getOptimalDFTSize(2*h-1), cv2.getOptimalDFTSize(2*w-1))
    centered = (feature - feature[mask > 0].mean()) * mask
    corr = _corr(centered, centered, shape)
    energy = np.sqrt(np.maximum(_corr(centered*centered, mask, shape) * _corr(mask, centered*centered, shape), 1e-9))
    overlap = _corr(mask, mask, shape)
    result = []
    for axis, limit in ((0, h), (1, w)):
        stop = int(limit * 0.7)
        x = np.arange(max(12, limit//18), stop)
        vals = corr[x, 0]/energy[x, 0] if axis == 0 else corr[0, x]/energy[0, x]
        ovs = overlap[x, 0] if axis == 0 else overlap[0, x]
        candidates = [(float(vals[i]), int(x[i])) for i in range(1, len(x)-1)
                      if vals[i] > 0.65 and vals[i] >= vals[i-1] and vals[i] >= vals[i+1]
                      and ovs[i] > mask.sum()*0.3]
        for strength, shift in sorted(candidates, reverse=True)[:2]:
            result.append({'vector_work_px': [shift if axis == 1 else 0, shift if axis == 0 else 0],
                           'autocorrelation': strength})
    return result


def _transform_difference(a, b, shape):
    h, w = shape
    pts = np.array([[[w*.2, h*.2], [w*.8, h*.2], [w*.5, h*.5], [w*.2, h*.8], [w*.8, h*.8]]], np.float32)
    return float(np.median(np.linalg.norm(cv2.perspectiveTransform(pts, a) - cv2.perspectiveTransform(pts, b), axis=2)))


def register(template, target, cfg=None, template_valid=None, target_valid=None, *, working_scale=None):
    """Return registration diagnostics; a high score is not an identity proof."""
    cfg = cfg or MatcherConfig()
    cfg.validate()
    a, b = to_gray(template), to_gray(target)
    am, bm = build_valid_mask(a, template_valid, cfg), build_valid_mask(b, target_valid, cfg)
    # Shared nominal scale, exact per-image rounding matrices retained.
    scale = min(1.0, cfg.max_side / max(*a.shape, *b.shape)) if working_scale is None else working_scale
    ashape = (round(a.shape[1]*scale), round(a.shape[0]*scale))
    bshape = (round(b.shape[1]*scale), round(b.shape[0]*scale))
    if min(*ashape, *bshape) < 32:
        raise ValueError('Extreme image aspect/size mismatch; use comparable fields of view')
    Da = np.diag([ashape[0]/a.shape[1], ashape[1]/a.shape[0], 1.0])
    Db = np.diag([bshape[0]/b.shape[1], bshape[1]/b.shape[0], 1.0])
    aa, am = ValidRegion.resize_for_matching(a, am, ashape)
    bb, bm = ValidRegion.resize_for_matching(b, bm, bshape)
    fail = dict(match_score=0.0, geometry_accepted=False, absolute_cell_verified=False,
                periodic_detected=False, matrix_template_to_target=None, candidates=[],
                working_scale=scale, working_target_size=list(bshape),
                reasons=['insufficient_texture_or_overlap'])
    if np.std(aa[am > 0]) < 2 or np.std(bb[bm > 0]) < 2:
        return fail
    src, dst = _features(aa, am), _features(bb, bm)
    if src[2].sum() < 40 or dst[2].sum() < 40:
        return fail
    lattice = _periods(dst[0], dst[3])
    coarse = []
    for angle in cfg.angles:
        for zoom in cfg.scales:
            A, size = _centered_warp(aa.shape, angle, zoom)
            sg = cv2.warpPerspective(src[1], A, size)
            sm = cv2.warpPerspective(src[3], A, size, flags=cv2.INTER_NEAREST)
            search = opencv_translation_peaks if cfg.backend == 'opencv' else _translation_peaks
            for merit, dx, dy in search(sg, dst[1], sm, dst[3], cfg):
                H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float) @ A
                coarse.append((merit, H))
    shortlist = []
    for merit, H in sorted(coarse, key=lambda x: x[0], reverse=True):
        if all(_transform_difference(H, prev, aa.shape) > 4 for _, prev in shortlist):
            shortlist.append((merit, H))
        if len(shortlist) >= cfg.refine_candidates:
            break
    candidates = []
    for _, H in shortlist:
        metrics = _score(src, dst, H, cfg)
        refined, ecc_ok = _refine(src, dst, H, cfg)
        refined_metrics = _score(src, dst, refined, cfg)
        if refined_metrics['match_score'] >= metrics['match_score']:
            H, metrics = refined, refined_metrics
        else:
            ecc_ok = False
        if all(_transform_difference(H, np.asarray(c['_H_work']), aa.shape) > 3 for c in candidates):
            metrics.update(_H_work=H.tolist(), ecc_converged=bool(ecc_ok),
                           matrix_template_to_target=(np.linalg.inv(Db) @ H @ Da).tolist())
            candidates.append(metrics)
    if not candidates:
        return fail
    candidates.sort(key=lambda c: c['match_score'], reverse=True)
    best = candidates[0].copy()
    gap = best['match_score'] - candidates[1]['match_score'] if len(candidates) > 1 else None
    # Autocorrelation plus far-apart near-equal matches flag periodic ambiguity.
    alias = any(best['match_score'] - c['match_score'] < cfg.min_candidate_gap and
                _transform_difference(np.asarray(best['_H_work']), np.asarray(c['_H_work']), aa.shape) > min(aa.shape)*0.12
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
                working_scale=scale, working_target_size=list(bshape), candidates=candidates,
                reasons=[] if accepted else ['score_or_coverage_below_threshold'])
    if periodic:
        best['reasons'].append('absolute_period_index_requires_external_anchor')
    return best


def select_templates(templates, target, cfg=None, template_valids=None, target_valid=None, label_mode='semantic'):
    """Rank templates on the same target ROI, scale and score definition.

    semantic allows shared cell classes; instance requires verified identity.
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
    cv2.setNumThreads(2)
    common_scale = min(1.0, cfg.max_side / max(max(to_gray(im).shape) for im in [*templates, target]))
    matches = []
    for i, (template, valid) in enumerate(zip(templates, template_valids)):
        result = register(template, target, cfg, valid, target_valid, working_scale=common_scale)
        result['template_index'] = i
        matches.append(result)
    ranked = sorted(matches, key=lambda r: r['match_score'], reverse=True)
    best = ranked[0]
    gap = best['match_score'] - ranked[1]['match_score'] if len(ranked) > 1 else None
    chosen = best['template_index'] if best['geometry_accepted'] else None
    separated = gap is None or gap >= cfg.min_template_gap
    visual_ok = bool(chosen is not None and separated)
    absolute_ok = bool(visual_ok and best['absolute_cell_verified'])
    accepted = absolute_ok if label_mode == 'instance' else visual_ok
    status = 'accepted' if accepted else ('low_match_score' if chosen is None else
             'templates_too_close' if not separated else 'absolute_cell_unverified')
    return dict(best_template_index=chosen, accepted=accepted, status=status,
                label_mode=label_mode, template_score_gap=gap,
                absolute_mapping_accepted=absolute_ok, ranking=[r['template_index'] for r in ranked],
                matches=matches, config=asdict(cfg),
                score_definition='100*(0.55*structure_ncc+0.30*edge_alignment+0.15*local_consistency)*sqrt(target_coverage)',
                score_is_probability=False)
