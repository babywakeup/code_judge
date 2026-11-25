"""Verified periodic copies fill unknown labels and complete boundary foreground."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import cv2
import numpy as np

from ...config import MatcherConfig
from ...utils import ValidRegion, to_gray, warp_labels, warp_valid_mask
from ...utils.types import Array, BoolMask, LabelMap, Mask
from ..registration.core import _features
from ..registration.scoring import ScoreWarp, _refine, _score, prepare_refinement_target
from ..models import MatchResult
from .lattice import calibrate_lattice
from .layers import projected_components, record_periodic_prediction
from .regions import RegionResize, component_regions


def anchor_vectors(labels: LabelMap, valid: Mask, class_id: int) -> list[Array]:
    """Estimate near-axis periods from repeated connected anchor components.

    Skewed/rotated lattices can instead use explicit period_vectors. Image
    autocorrelation remains a fallback when anchor components are insufficient.
    """
    count, _, stats, centers = cv2.connectedComponentsWithStats(
        ((labels == class_id) & (valid > 0)).astype(np.uint8))
    centers = centers[1:][stats[1:, cv2.CC_STAT_AREA] >= 16]
    if len(centers) < 2:
        return []
    vectors = []
    for axis in (0, 1):
        differences = []
        for a in centers:
            for b in centers:
                delta = b-a
                if delta[axis] >= 12 and abs(delta[1-axis]) <= .15*delta[axis]:
                    differences.append(delta)
        if not differences:
            continue
        differences = np.asarray(differences)
        minimum = differences[:, axis].min()
        cluster = differences[differences[:, axis] <= minimum*1.15]
        if len(cluster) >= 2:  # Require repetition, not a single accidental pair.
            vectors.append(np.median(cluster, axis=0))
    return vectors


def _candidate_fill(mapped: LabelMap, annotation: BoolMask, target: Mask,
                    labels: LabelMap, known: BoolMask) -> BoolMask:
    """Include whole foreground objects crossing from unknown into background.

    Only components with foreground pixels in an unknown target region can seed
    an extension. Existing nonzero labels remain protected, and unrelated objects
    entirely inside the direct region cannot expand just because they repeat.
    """
    eligible = annotation & (target > 0)
    unknown = eligible & ~known
    fill = unknown.copy()
    seeds = unknown & (mapped > 0)
    for class_id in np.unique(mapped[seeds]):
        class_mask = mapped == class_id
        count, components = cv2.connectedComponents((class_mask & eligible).astype(np.uint8))
        seeded = np.zeros(count, bool)
        seeded[components[seeds & class_mask]] = True
        seeded[0] = False
        fill |= seeded[components] & known & (labels == 0)
    return fill


def complete_periodic(result: MatchResult, labels: LabelMap, annotation_valid: Mask,
                      source_valid: Mask, target_valid: Mask, cfg: MatcherConfig) -> None:
    source, target = result.template_image, result.input_image
    H = result.matrix_template_to_target
    h, w = target.shape[:2]
    best = result.registration['matches'][result.registration['ranking'][0]]
    low_frequency = best.get('evidence_mode') == 'low_frequency'
    if low_frequency:
        # Filtered edges are coarser and fewer. Require stronger structure on
        # this path, which is entered only after filtered registration passes.
        cfg = replace(cfg, periodic_min_ncc=max(cfg.periodic_min_ncc, cfg.periodic_low_frequency_min_ncc),
                      periodic_min_edge=min(cfg.periodic_min_edge, cfg.periodic_low_frequency_edge_cap))
    scale = cfg.matching_scale(source.shape, target.shape)
    ss = (round(source.shape[1]*scale), round(source.shape[0]*scale))
    ts = (round(w*scale), round(h*scale))
    resize_region = RegionResize((h, w), ts)
    Da = np.diag([ss[0]/source.shape[1], ss[1]/source.shape[0], 1.])
    Db = np.diag([ts[0]/w, ts[1]/h, 1.])

    def image_vectors() -> list[Array]:
        inferred = []
        working_size = best.get('working_target_size', ts)
        for axis in (0, 1):
            candidates = [v for v in best.get('lattice_candidates', [])
                          if v['vector_work_px'][axis] > 0 and v['vector_work_px'][1-axis] == 0]
            if candidates:
                candidate = min(candidates, key=lambda v: v['vector_work_px'][axis])
                target_vector = np.asarray(candidate['vector_work_px']) * [w/working_size[0], h/working_size[1]]
                inferred.append(np.linalg.solve(H[:2, :2], target_vector))
        return inferred

    vectors = [np.asarray(v, float) for v in cfg.period_vectors]
    period_source = 'configured'
    if not vectors:
        vectors = anchor_vectors(labels, annotation_valid, cfg.periodic_anchor_class)
        period_source = 'anchor_class'
    if not vectors:
        period_source = 'image_autocorrelation'
        vectors = image_vectors()
    if result.completed_template is None:
        result.completed_template = result.warped_template.copy()
        result.completed_valid = result.geometric_valid.copy()
    result.periodic_valid = np.zeros((h, w), bool)
    result.label_periodic_valid = np.zeros((h, w), bool)
    total = int(target_valid.sum())
    info = dict(status='incomplete', period_source=period_source,
                period_vectors=[v.tolist() for v in vectors], copies=[],
                direct_label_pixels=int(result.label_valid.sum()), target_valid_pixels=total)
    if low_frequency:
        info['evidence_mode'] = 'low_frequency'
    # The ordered class has its own identity-aware validity; it must not block
    # independent base background. Every other declared/observed class matters.
    policy_classes = ({int(c) for c in np.unique(labels) if c > 0}
                      | {int(c) for c in cfg.class_periodicity}) - {cfg.ordered_object_class}
    info['class_periodicity'] = {str(c): list(cfg.periodic_axes(c)) for c in sorted(policy_classes)}
    result.registration['periodic_completion'] = info

    def finish(status: str | None = None) -> None:
        missing = int((target_valid & ~result.label_valid).sum())
        info.update(status=status or ('complete' if missing == 0 else 'incomplete'),
                    label_coverage_roi=float(result.label_valid.sum()/total),
                    geometric_coverage_roi=float(result.completed_valid.sum()/total),
                    added_label_pixels=int(result.label_periodic_valid.sum()),
                    newly_covered_pixels=int(result.label_valid.sum()) - info['direct_label_pixels'],
                    boundary_foreground_pixels=(int(result.label_periodic_valid.sum())
                                                - int(result.label_valid.sum()) + info['direct_label_pixels']),
                    remaining_unknown_pixels=missing)

    # Base labels may already cover the ROI while a separately localized object
    # still lacks support. Image geometry must not depend on base-label holes.
    needs_object_geometry = (cfg.ordered_object_class is not None
                             and cfg.object_repeat_axis in cfg.periodic_axes(cfg.ordered_object_class))
    if result.label_valid[target_valid].all() and not needs_object_geometry:
        finish()
        return
    if not vectors:
        finish('no_period')
        return
    source_gray, source_mask = ValidRegion.resize_for_matching(to_gray(source), source_valid, ss)
    target_gray, target_mask = ValidRegion.resize_for_matching(to_gray(target), target_valid, ts)
    src = _features(source_gray, source_mask, low_frequency=low_frequency, cfg=cfg)
    dst = _features(target_gray, target_mask, low_frequency=low_frequency, cfg=cfg)
    local_cfg = replace(cfg, min_overlap=cfg.local_min_overlap)
    if period_source == 'anchor_class':
        # One class may contain different objects inside a periodic cell. Their
        # closest centroid spacing is not necessarily a period of the photograph.
        validation = []
        supported = []
        for vector in vectors:
            translation = np.eye(3)
            translation[:2, 2] = Da[:2, :2] @ vector
            metrics = _score(src, src, translation, local_cfg)
            accepted = (metrics['structure_ncc'] >= cfg.periodic_min_ncc
                        and metrics['edge_alignment'] >= cfg.periodic_min_edge)
            validation.append(dict(vector=vector.tolist(), accepted=bool(accepted), metrics=metrics))
            if accepted:
                supported.append(vector)
        info['anchor_period_validation'] = validation
        if len(supported) != len(vectors):
            fallback = image_vectors()
            vectors = supported.copy()
            for vector in fallback:
                # Keep validated anchor axes when autocorrelation finds only one
                # direction; do not add a parallel harmonic as a second axis.
                if all(abs(vector[0]*prev[1]-vector[1]*prev[0]) >
                       .15*np.linalg.norm(vector)*np.linalg.norm(prev) for prev in vectors):
                    vectors.append(vector)
                if len(vectors) == 2:
                    break
            period_source = 'image_autocorrelation' if len(vectors) > len(supported) else 'anchor_class'
            info.update(period_source=period_source, period_vectors=[v.tolist() for v in vectors],
                        period_fallback_reason='anchor_spacing_not_supported_by_image')
        if not vectors:
            finish('no_period')
            return
    lattice = None
    if cfg.shared_periods:
        lattice = calibrate_lattice(src, dst, vectors, H, Da, Db, cfg,
                                    configured=period_source == 'configured',
                                    objects=result.registration.get('ordered_objects'))
        vectors = [np.asarray(axis['source_vector']) for axis in lattice]
        info.update(period_vectors=[v.tolist() for v in vectors], lattice=lattice,
                    shared_geometry_copies=0, copy_refinement_calls=0)
    offsets = [v for v in product(range(-cfg.periodic_max_steps, cfg.periodic_max_steps+1), repeat=len(vectors))
               if any(v)]
    offsets.sort(key=lambda v: (sum(abs(x) for x in v), v))
    refinement = None
    for offset in offsets:
        active_vectors = [v for n, v in zip(offset, vectors) if n]
        shift = sum((n*v for n, v in zip(offset, vectors)), np.zeros(2))
        P = np.eye(3)
        P[:2, 2] = shift
        candidate = H @ P
        shared = lattice is not None and all(axis['verified'] for n, axis in zip(offset, lattice) if n)
        if shared:
            target_shift = sum((n*np.asarray(axis['target_vector']) for n, axis in zip(offset, lattice)), np.zeros(2))
            candidate = H.copy()
            candidate[:2, 2] += target_shift
        # Skip candidates unable to supply any currently unknown annotation.
        has_unknown = not result.label_valid[target_valid].all()
        needs_geometry = needs_object_geometry
        if cfg.shared_periods:
            needs_geometry = (needs_object_geometry
                              and cfg.allows_periodic_shift(cfg.ordered_object_class, active_vectors))
        ann = (warp_valid_mask(annotation_valid, candidate, target.shape)
               if has_unknown or not cfg.shared_periods else None)
        supplies_labels = bool(ann is not None and (ann & target_valid & ~result.label_valid).any())
        if not supplies_labels and not needs_geometry:
            continue
        work = Db @ candidate @ np.linalg.inv(Da)
        initial_warp = ScoreWarp(src, work, ts)
        initial_metrics = _score(src, dst, work, local_cfg, warped=initial_warp)
        if shared:
            # Geometry is H plus a verified integer lattice displacement. Do
            # not independently bend every x/y/diagonal copy with affine ECC.
            refined, ecc_ok = work, False
            info['shared_geometry_copies'] += 1
        else:
            if refinement is None:
                refinement = prepare_refinement_target(dst)
            refined, ecc_ok = _refine(src, dst, work, local_cfg, prepared=refinement, initial_warp=initial_warp)
            if lattice is not None:
                info['copy_refinement_calls'] += 1
        refined_warp = initial_warp if np.array_equal(refined, work) else ScoreWarp(src, refined, ts)
        # Prevent ECC from jumping to another cell or destroying the lattice.
        points = np.array([[[ss[0]*.2, ss[1]*.2], [ss[0]*.8, ss[1]*.8], [ss[0]*.5, ss[1]*.5]]], np.float32)
        displacement = np.linalg.norm(cv2.perspectiveTransform(points, work)-cv2.perspectiveTransform(points, refined), axis=2)
        limit = min(16., min(np.linalg.norm(Db[:2, :2] @ H[:2, :2] @ v) for v in vectors)*.1)
        within_limit = bool(displacement.max() <= limit)
        refined_metrics = (initial_metrics if refined_warp is initial_warp or not within_limit else
                   _score(src, dst, refined, local_cfg, warped=refined_warp))
        if within_limit and refined_metrics['match_score'] >= initial_metrics['match_score']:
            candidate = np.linalg.inv(Db) @ refined @ Da
            metrics = refined_metrics
            selected_warp = refined_warp
        else:
            metrics, ecc_ok = initial_metrics, False
            selected_warp = initial_warp
        accepted = metrics['structure_ncc'] >= cfg.periodic_min_ncc and metrics['edge_alignment'] >= cfg.periodic_min_edge
        record = dict(offset=list(offset), accepted=bool(accepted), metrics=metrics,
                      geometry_accepted=bool(accepted), ecc_converged=bool(ecc_ok),
                      matrix_template_to_target=candidate.tolist(), added_label_pixels=0)
        if lattice is not None:
            record['refinement_mode'] = 'shared_lattice' if shared else 'independent_fallback'
        info['copies'].append(record)
        if not accepted:
            continue
        if cfg.shared_periods and not supplies_labels:
            record['accepted'] = False
            continue
        mapped, ann = warp_labels(labels, candidate, target.shape, annotation_valid)
        allowed = {c for c in policy_classes if cfg.allows_periodic_shift(c, active_vectors)}
        blocked = policy_classes - allowed
        record['blocked_class_ids'] = sorted(blocked)
        # Never turn a prohibited foreground copy into background. Moreover,
        # translated background cannot prove a nonperiodic class is absent.
        # Keep such negative knowledge unknown; approved foreground of another
        # class can still establish a positive label at its actual source pixels.
        eligible = (mapped == next(iter(allowed)) if len(allowed) == 1
                    else np.isin(mapped, sorted(allowed)))
        if not blocked:
            eligible |= mapped == 0
        ann &= eligible
        fill_labels = _candidate_fill(mapped, ann, target_valid, result.mapped_labels, result.label_valid)
        # Validate each new connected region, not only the large aligned interior.
        # A bounded context supplies edges for thin strips and corner fragments.
        approved = np.zeros((h, w), bool)
        record['local_regions'] = []
        candidate_work = Db @ candidate @ np.linalg.inv(Da)
        score_warp = (selected_warp if np.array_equal(candidate_work, selected_warp.H)
                      else ScoreWarp(src, candidate_work, ts))
        for region in component_regions(fill_labels):
            local = resize_region(region)
            local = cv2.dilate(local, np.ones((cfg.local_context_size,)*2, np.uint8)) & dst[3]
            local_dst = (dst[0], dst[1], dst[2], local)
            local_metrics = _score(src, local_dst, candidate_work, local_cfg, warped=score_warp,
                                   min_structure_ncc=cfg.periodic_min_ncc)
            local_ok = (local_metrics['structure_ncc'] >= cfg.periodic_min_ncc
                        and local_metrics['edge_alignment'] >= cfg.periodic_min_edge)
            record['local_regions'].append(dict(accepted=bool(local_ok), pixels=region.pixels, metrics=local_metrics))
            if local_ok:
                approved[region.slices] |= region.mask
        fill_labels = approved
        record['accepted'] = bool(fill_labels.any())
        if not fill_labels.any():
            continue
        record_periodic_prediction(result, mapped,
                                   projected_components(mapped, ann & target_valid, approved))
        record['newly_covered_pixels'] = int((fill_labels & ~result.label_valid).sum())
        record['boundary_foreground_pixels'] = int((fill_labels & result.label_valid).sum())
        result.mapped_labels[fill_labels] = mapped[fill_labels]
        result.label_valid |= fill_labels
        result.label_periodic_valid |= fill_labels
        fill_image = approved & ~result.completed_valid
        if fill_image.any():
            fill_image &= warp_valid_mask(source_valid, candidate, target.shape)
            if fill_image.any():
                warped = cv2.warpPerspective(source, candidate, (w, h))
                result.completed_template[fill_image] = warped[fill_image]
                result.completed_valid |= fill_image
                result.periodic_valid |= fill_image
        record['added_label_pixels'] = int(fill_labels.sum())
        if result.label_valid[target_valid].all() and not (cfg.shared_periods and needs_object_geometry):
            break
    finish()
