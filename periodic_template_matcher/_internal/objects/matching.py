"""Identity-aware appearance matching for cyclic groups of different objects.

No label map is periodically tiled here. Each candidate comes from an original
object reference and actual visible target pixels; order only resolves candidates.
"""
from collections.abc import Sequence
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils import ValidRegion, to_gray, warp_valid_mask
from ...utils.types import BoolMask, Diagnostics, Image, LabelMap, Mask, Matrix
from ..registration.core import _features
from ..registration.scoring import (ScoreWarp, _refine, _score, prepare_refinement_target,
                                    refinement_for_support)
from .appearance import _correlation_peaks
from .axes import transpose_image, transpose_matrix, transpose_config, transpose_prediction
from .layout import _quality, _template_spacings, is_partial_reference, shared_foreground, solve_layout


@dataclass
class ObjectPrediction:
    """Target-sized class IDs, cyclic identity IDs and independently known pixels."""

    labels: LabelMap
    identities: LabelMap
    valid: BoolMask
    diagnostics: Diagnostics
    semantic_foreground: BoolMask | None = None


def _refinement_features(image, source_valid, target, target_valid, cfg, *, low_frequency=False):
    """Invariant working images for one source/target pair."""
    scale = cfg.matching_scale(target.shape, image.shape)
    ss = (round(image.shape[1]*scale), round(image.shape[0]*scale))
    ts = (round(target.shape[1]*scale), round(target.shape[0]*scale))
    Da = np.diag([ss[0]/image.shape[1], ss[1]/image.shape[0], 1.])
    Db = np.diag([ts[0]/target.shape[1], ts[1]/target.shape[0], 1.])
    sg, sm = ValidRegion.resize_for_matching(to_gray(image), source_valid, ss)
    tg, tm = ValidRegion.resize_for_matching(to_gray(target), target_valid, ts)
    src = _features(sg, sm, low_frequency=low_frequency, cfg=cfg)
    dst = _features(tg, tm, low_frequency=low_frequency, cfg=cfg)
    return ts, Da, Db, src, dst, tm


def refine_object_transform(image, mask, source_valid, target, target_valid, matrix, cfg, *,
                            _prepared=None, _target_refinement=None, _proposal=None):
    """Refine an already identified occurrence, bounded to its local neighborhood."""
    ts, Da, Db, src, dst, tm = (_refinement_features(image, source_valid, target, target_valid, cfg)
                               if _prepared is None else _prepared)
    scale = cfg.matching_scale(target.shape, image.shape)
    observed = warp_valid_mask(mask, matrix, target.shape)
    local = cv2.resize(observed.astype(np.uint8), ts, interpolation=cv2.INTER_NEAREST)
    local = cv2.dilate(local, np.ones((9, 9), np.uint8)) & tm
    dst = (*dst[:3], local)
    work = Db @ matrix @ np.linalg.inv(Da)
    config = replace(cfg, min_overlap=cfg.local_min_overlap)
    proposal = ScoreWarp(src, work, ts)
    before = _score(src, dst, work, config, warped=proposal)
    refinement = (None if _target_refinement is None
                  else refinement_for_support(_target_refinement, local))
    if _proposal is None:
        refined, converged = _refine(src, dst, work, config, prepared=refinement, initial_warp=proposal)
    else:
        # A verified alias offers a transform, not an acceptance exemption.
        # All original local score/displacement conditions below still apply.
        refined, converged = Db @ _proposal @ np.linalg.inv(Da), True
    if not converged or np.array_equal(work, refined):
        return matrix
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return matrix
    points = np.array([[[xs.min(), ys.min()], [xs.max(), ys.min()],
                        [xs.max(), ys.max()], [xs.min(), ys.max()]]], np.float32)
    candidate = np.linalg.inv(Db) @ refined @ Da
    displacement = np.linalg.norm(cv2.perspectiveTransform(points, candidate)
                                  - cv2.perspectiveTransform(points, matrix), axis=2)
    # A rejected geometric proposal cannot be rescued by its score. Preserve
    # the exact <= test (including NaN rejection) before expensive edge scoring.
    if not displacement.max()*scale <= 6:
        return matrix
    after = _score(src, dst, refined, config)
    if (after['structure_ncc'] > before['structure_ncc'] + .001
            and after['edge_alignment'] >= before['edge_alignment']
            and after['match_score'] > before['match_score']):
        return candidate
    return matrix


def _reference_hypotheses(images, references, source_valids, matrices, cfg):
    """All source references may offer locally verifiable appearance evidence.

    In partial-cycle mode, unregistered sources use configured nominal poses;
    their global translation/score is never treated as local identity evidence.
    """
    for ti, (image, refs, valid, matrix) in enumerate(zip(images, references, source_valids, matrices)):
        if not refs:
            continue
        if matrix is not None:
            yield ti, image, refs, valid, matrix
        elif cfg.object_visible_count is not None:
            for angle in cfg.angles:
                for scale in cfg.scales:
                    nominal = np.vstack([cv2.getRotationMatrix2D((0., 0.), float(angle), float(scale)), [0., 0., 1.]])
                    yield ti, image, refs, valid, nominal


def match_ordered_objects(
    images: Sequence[Image], references: Sequence[Sequence[Diagnostics]],
    source_valids: Sequence[Mask], target: Image, target_valid: Mask,
    matrices: Sequence[Matrix | Sequence[Sequence[float]] | None], cfg: MatcherConfig,
) -> ObjectPrediction:
    """Match a visible cyclic window along the configured x/y layout axis."""
    prediction = _match_ordered_objects(images, references, source_valids, target, target_valid, matrices, cfg)
    if (cfg.robust_appearance and any(matrix is not None for matrix in matrices)
            and prediction.diagnostics['status'] == 'insufficient_evidence'):
        fallback = _match_ordered_objects(images, references, source_valids, target, target_valid,
                                          matrices, cfg, low_frequency=True)
        if fallback.diagnostics['status'] in ('accepted', 'partial_reference'):
            fallback.diagnostics.update(evidence_mode='low_frequency',
                                        original_candidate_counts=prediction.diagnostics['candidate_counts'])
            return fallback
    return prediction


def _match_ordered_objects(images, references, source_valids, target, target_valid, matrices, cfg, *,
                           low_frequency=False):
    if cfg.object_order_axis == 'y':
        prediction = _match_ordered_objects([transpose_image(im) for im in images],
            [[dict(r, mask=transpose_image(r['mask'])) for r in refs] for refs in references],
            [transpose_image(v) for v in source_valids], transpose_image(target), transpose_image(target_valid),
            [transpose_matrix(m) for m in matrices], transpose_config(cfg), low_frequency=low_frequency)
        return transpose_prediction(prediction)
    shape = target.shape[:2]
    labels = np.zeros(shape, np.uint16)
    identities = np.zeros(shape, np.uint16)
    known = np.zeros(shape, bool)
    cycle = list(cfg.object_cycle)
    info = dict(status='insufficient_evidence', visible_order='', identity_names={str(i+1): n for i, n in enumerate(cycle)},
                class_id=cfg.ordered_object_class, order_axis='x',
                objects=[], candidate_counts={}, alternatives=[], valid_coverage_roi=0.)
    scale = cfg.matching_scale(target.shape, *(im.shape for im in images))
    ts = (round(shape[1]*scale), round(shape[0]*scale))
    Db = np.diag([ts[0]/shape[1], ts[1]/shape[0], 1.])
    tg, tm = ValidRegion.resize_for_matching(to_gray(target), target_valid, ts)
    target_features = _features(tg, tm, low_frequency=low_frequency, cfg=cfg)
    tf = target_features[0]
    target_roi = target_valid > 0
    source_features = {}
    source_masks = {}
    refinement_features = {}
    refinement_targets = {}
    pools = {name: [] for name in cycle}
    for ti, image, refs, source_valid, initial in _reference_hypotheses(
            images, references, source_valids, matrices, cfg):
        ss = (round(image.shape[1]*scale), round(image.shape[0]*scale))
        Da = np.diag([ss[0]/image.shape[1], ss[1]/image.shape[0], 1.])
        if ti not in source_features:
            sg, sm = ValidRegion.resize_for_matching(to_gray(image), source_valid, ss)
            source_features[ti] = _features(sg, sm, low_frequency=low_frequency, cfg=cfg)
            source_masks[ti] = sm
            # Local refinement historically uses the pair's matching scale.
            # Reuse only when identical; differently sized templates must keep
            # their original pairwise resolution and coordinate matrices.
            if cfg.matching_scale(target.shape, image.shape) == scale:
                refinement_features[ti] = ts, Da, Db, source_features[ti], target_features, tm
        # Feature extraction erodes its safety mask. Appearance visibility
        # and local context need the original resized ROI.
        sf, sm = source_features[ti][0], source_masks[ti]
        linear = Db @ np.asarray(initial) @ np.linalg.inv(Da)
        linear[:2, 2] = 0
        corners = cv2.perspectiveTransform(np.array([[[0., 0.], [ss[0]-1., 0.],
                           [ss[0]-1., ss[1]-1.], [0., ss[1]-1.]]], np.float32), linear)[0]
        low, high = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
        linear[:2, 2] -= low
        canvas_size = tuple((high-low+1).astype(int))
        aligned = cv2.warpPerspective(sf, linear, canvas_size)
        aligned_valid = cv2.warpPerspective(sm, linear, canvas_size, flags=cv2.INTER_NEAREST)
        for ri, ref in enumerate(refs):
            raw = ((ref['mask'] > 0) & (source_valid > 0)).astype(np.uint8)
            if not raw.any():
                continue
            mask = cv2.resize(raw, ss, interpolation=cv2.INTER_NEAREST)
            mask = cv2.warpPerspective(mask, linear, canvas_size, flags=cv2.INTER_NEAREST)
            context = cv2.dilate(mask, np.ones((9, 9), np.uint8)) & aligned_valid
            yy, xx = np.where(context > 0)
            if len(xx) < cfg.object_min_work_pixels:
                continue
            x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
            patch, weight = aligned[y0:y1, x0:x1], context[y0:y1, x0:x1]
            raw_ys, raw_xs = np.where(raw > 0)
            raw_pixels = float(len(raw_xs))
            raw_height = raw_ys.max()-raw_ys.min()+1
            reference_partial = is_partial_reference(ref, int(raw_xs.min()), int(raw_xs.max()), image.shape[1])
            center = np.array([[[float((raw_xs.min()+raw_xs.max())/2),
                                float((raw_ys.min()+raw_ys.max())/2)]]], np.float32)
            for score, dx, dy in _correlation_peaks(patch, weight, tf, tm, cfg):
                shift = np.eye(3)
                shift[:2, 2] = [dx-x0, dy-y0]
                H = np.linalg.inv(Db) @ shift @ linear @ Da
                visible = warp_valid_mask(raw, H, target.shape) & target_roi
                visible_pixels = visible.sum()
                fraction = min(1., float(visible_pixels/max(raw_pixels*abs(np.linalg.det(H[:2, :2])), 1)))
                if fraction < cfg.object_min_visible or visible_pixels < cfg.min_contour_pixels:
                    continue
                vertical_fraction = min(1., float(np.any(visible, axis=1).sum()/max(
                    raw_height*np.linalg.norm(H[:2, 1]), 1)))
                cx = float(cv2.perspectiveTransform(center, H)[0, 0, 0])
                pools[ref['identity']].append(dict(identity=ref['identity'], template_index=ti,
                    reference_index=ri, shape_index=ref.get('shape_index', ri), score=score,
                    reference_partial=bool(reference_partial),
                    visible_fraction=fraction, partial=fraction < .95, center_x=cx,
                    vertical_fraction=vertical_fraction,
                    horizontal_fraction=min(1., fraction/max(vertical_fraction, 1e-6)),
                    matrix_template_to_target=H.tolist(), _mask=visible,
                    _background_known=ref.get('background_known', True)))
    # Partial references can establish spacing through locally verified poses,
    # even if their whole-image registration was rejected.
    spacing_matrices = list(matrices)
    if cfg.object_visible_count is not None:
        for ti in range(len(images)):
            local = [c for candidates in pools.values() for c in candidates if c['template_index'] == ti]
            if local:
                spacing_matrices[ti] = max(local, key=_quality)['matrix_template_to_target']
    info['reference_counts'] = {str(i): len(refs) for i, refs in enumerate(references)}
    info['candidate_counts_by_template'] = {str(i): sum(c['template_index'] == i for candidates in pools.values()
                                                       for c in candidates) for i in range(len(images))}
    for name in cycle:
        candidates = sorted(pools[name], key=_quality, reverse=True)
        distinct = []
        for candidate in candidates:
            # Retain one best reference per horizontal location; vertical variants
            # are kept separately for same-identity support after order resolution.
            if all(abs(candidate['center_x']-prev['center_x']) > max(8., 12/scale) for prev in distinct):
                distinct.append(candidate)
            if len(distinct) >= cfg.object_candidates:
                break
        pools[name] = dict(all=candidates, distinct=distinct)
        info['candidate_counts'][name] = len(candidates)
    spacing = _template_spacings(images, references, spacing_matrices, cycle)
    info['layout_spacings'] = [dict(left=a, right=b, target_px=v) for (a, b), v in spacing.items()]
    info['cycle_size'] = len(cycle)
    info['instance_count'] = cfg.object_visible_count or len(cycle)
    groups = solve_layout(pools, cycle, spacing, cfg.object_spacing_weight, cfg.object_visible_count)
    if not groups:
        return ObjectPrediction(labels, identities, known, info)
    score, order, chosen = groups[0]
    alternative = next((g for g in groups[1:] if g[1] != order or len(cycle) == 1), None)
    margin = score-alternative[0] if alternative is not None else None
    info.update(visible_order=''.join(order), visible_identities=order, score=score, order_margin=margin,
                alternatives=[dict(order=''.join(o), score=s) for s, o, _ in groups[:6]],
                proposed_objects=[{k: v for k, v in c.items() if not k.startswith('_')} for c in chosen])
    if margin is not None and margin < cfg.object_min_gap:
        info['status'] = 'ambiguous_identity'
        semantic = None
        if cfg.object_visible_count is not None:
            semantic = shared_foreground(groups, cfg.object_min_gap)
            info['semantic_consensus_pixels'] = int(semantic.sum())
        return ObjectPrediction(labels, identities, known, info, semantic)
    info['status'] = 'accepted'
    if cfg.shared_periods:
        info.update(periodic_refinement_reuses=0, periodic_refinement_fallbacks=0)
    known = target_valid > 0
    for chosen_object in chosen:
        name = chosen_object['identity']
        selected = [c for c in pools[name]['all'] if abs(c['center_x']-chosen_object['center_x']) < 8/scale
                    and c['score'] >= chosen_object['score']-.025]
        combined = np.zeros(shape, bool)
        support = np.zeros(shape, bool)
        periodic_refinements = {}
        for candidate in selected:
            ti = candidate['template_index']
            if cfg.object_visible_count is not None:
                ti, ri = candidate['template_index'], candidate['reference_index']
                raw = ((references[ti][ri]['mask'] > 0) & (source_valids[ti] > 0)).astype(np.uint8)
                initial = np.asarray(candidate['matrix_template_to_target'])
                if ti not in refinement_features:
                    refinement_features[ti] = _refinement_features(
                        images[ti], source_valids[ti], target, target_valid, cfg, low_frequency=low_frequency)
                pair_features = refinement_features[ti]
                target_key = id(pair_features[4][0])
                if target_key not in refinement_targets:
                    refinement_targets[target_key] = prepare_refinement_target(pair_features[4])
                refined = initial
                alias_key = ti, ri
                if cfg.shared_periods and alias_key in periodic_refinements:
                    anchor, adjusted = periodic_refinements[alias_key]
                    if np.array_equal(anchor[:2, :2], initial[:2, :2]):
                        source_shift = np.linalg.solve(anchor[:2, :2], initial[:2, 2]-anchor[:2, 2])
                        if np.linalg.norm(source_shift)*scale >= 12 and cfg.allows_periodic_shift(
                                cfg.ordered_object_class, [source_shift]):
                            translation = np.eye(3)
                            translation[:2, 2] = pair_features[1][:2, :2] @ source_shift
                            evidence = _score(pair_features[3], pair_features[3], translation,
                                              replace(cfg, min_overlap=cfg.local_min_overlap))
                            if (evidence['structure_ncc'] >= cfg.periodic_min_ncc
                                    and evidence['edge_alignment'] >= cfg.periodic_min_edge):
                                shared = adjusted @ np.linalg.inv(anchor) @ initial
                                refined = refine_object_transform(images[ti], raw, source_valids[ti], target,
                                    target_valid, initial, cfg, _prepared=pair_features,
                                    _target_refinement=refinement_targets[target_key], _proposal=shared)
                                if not np.array_equal(initial, refined):
                                    info['periodic_refinement_reuses'] += 1
                if np.array_equal(initial, refined):
                    refined = refine_object_transform(images[ti], raw, source_valids[ti], target,
                        target_valid, initial, cfg, _prepared=pair_features,
                        _target_refinement=refinement_targets[target_key])
                    if cfg.shared_periods:
                        info['periodic_refinement_fallbacks'] += 1
                if cfg.shared_periods and not np.array_equal(initial, refined) and alias_key not in periodic_refinements:
                    periodic_refinements[alias_key] = initial.copy(), refined.copy()
                candidate['local_refined'] = not np.array_equal(initial, refined)
                candidate['matrix_template_to_target'] = refined.tolist()
                yy, xx = np.where(raw > 0)
                center = np.array([[[float((xx.min()+xx.max())/2),
                                    float((yy.min()+yy.max())/2)]]], np.float32)
                candidate['center_x'] = float(cv2.perspectiveTransform(center, refined)[0, 0, 0])
                if candidate['local_refined']:
                    candidate['_mask'] = warp_valid_mask(raw, refined, target.shape) & target_roi
            combined |= candidate['_mask']
            # Layout selection never uses source background support. Project it
            # only for the selected occurrences, after any local refinement.
            support |= warp_valid_mask(source_valids[ti], candidate['matrix_template_to_target'], target.shape)
        # Missing vertical source support cannot be invented at target boundaries.
        ys, xs = np.where(combined)
        band = np.zeros(shape, bool)
        band[:, xs.min():xs.max()+1] = True
        known &= ~(band & ~support)
        if not chosen_object['_background_known']:
            known &= combined
        if all(c['reference_partial'] for c in selected):
            # A fragment says where observed foreground is, not how far the
            # unseen object continues beyond the source photograph.
            known &= combined
            info['status'] = 'partial_reference'
        labels[combined] = cfg.ordered_object_class
        identities[combined] = cycle.index(name)+1
        record = {k: v for k, v in chosen_object.items() if not k.startswith('_')}
        record['supporting_transforms'] = [c['matrix_template_to_target'] for c in selected]
        record['supporting_references'] = [dict(template_index=c['template_index'], reference_index=c['reference_index'],
            matrix_template_to_target=c['matrix_template_to_target']) for c in selected]
        info['objects'].append(record)
    known |= labels > 0
    known &= target_valid > 0
    info['valid_coverage_roi'] = float(known.sum()/max(int((target_valid > 0).sum()), 1))
    return ObjectPrediction(labels, identities, known, info)
