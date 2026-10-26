"""Verified lattice copies fill unknown target labels without overwriting direct data."""
from dataclasses import replace
from itertools import product

import cv2
import numpy as np

from .registration import _features, _refine, _score
from ..utils import ValidRegion, to_gray, warp_labels


def anchor_vectors(labels, valid, class_id):
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


def complete_periodic(result, labels, annotation_valid, source_valid, target_valid, cfg):
    source, target = result.template_image, result.input_image
    H = result.matrix_template_to_target
    h, w = target.shape[:2]
    best = result.registration['matches'][result.registration['ranking'][0]]
    scale = min(1., cfg.max_side/max(*source.shape[:2], h, w))
    ss = (round(source.shape[1]*scale), round(source.shape[0]*scale))
    ts = (round(w*scale), round(h*scale))
    Da = np.diag([ss[0]/source.shape[1], ss[1]/source.shape[0], 1.])
    Db = np.diag([ts[0]/w, ts[1]/h, 1.])
    vectors = [np.asarray(v, float) for v in cfg.period_vectors]
    period_source = 'configured'
    if not vectors:
        vectors = anchor_vectors(labels, annotation_valid, cfg.periodic_anchor_class)
        period_source = 'anchor_class'
    if not vectors:
        period_source = 'image_autocorrelation'
        working_size = best.get('working_target_size', ts)
        for axis in (0, 1):
            candidates = [v for v in best.get('lattice_candidates', [])
                          if v['vector_work_px'][axis] > 0 and v['vector_work_px'][1-axis] == 0]
            if candidates:
                candidate = min(candidates, key=lambda v: v['vector_work_px'][axis])
                target_vector = np.asarray(candidate['vector_work_px']) * [w/working_size[0], h/working_size[1]]
                vectors.append(np.linalg.solve(H[:2, :2], target_vector))
    result.completed_template = result.warped_template.copy()
    result.completed_valid = result.geometric_valid.copy()
    result.periodic_valid = np.zeros((h, w), bool)
    result.label_periodic_valid = np.zeros((h, w), bool)
    total = int(target_valid.sum())
    info = dict(status='incomplete', period_source=period_source,
                period_vectors=[v.tolist() for v in vectors], copies=[],
                direct_label_pixels=int(result.label_valid.sum()), target_valid_pixels=total)
    result.registration['periodic_completion'] = info

    def finish(status=None):
        missing = int((target_valid & ~result.label_valid).sum())
        info.update(status=status or ('complete' if missing == 0 else 'incomplete'),
                    label_coverage_roi=float(result.label_valid.sum()/total),
                    geometric_coverage_roi=float(result.completed_valid.sum()/total),
                    added_label_pixels=int(result.label_periodic_valid.sum()),
                    remaining_unknown_pixels=missing)

    if result.label_valid[target_valid].all():
        finish()
        return
    if not vectors:
        finish('no_period')
        return
    source_gray, source_mask = ValidRegion.resize_for_matching(to_gray(source), source_valid, ss)
    target_gray, target_mask = ValidRegion.resize_for_matching(to_gray(target), target_valid, ts)
    src = _features(source_gray, source_mask)
    dst = _features(target_gray, target_mask)
    local_cfg = replace(cfg, min_overlap=.2)
    offsets = [v for v in product(range(-cfg.periodic_max_steps, cfg.periodic_max_steps+1), repeat=len(vectors))
               if any(v)]
    offsets.sort(key=lambda v: (sum(abs(x) for x in v), v))
    for offset in offsets:
        shift = sum((n*v for n, v in zip(offset, vectors)), np.zeros(2))
        P = np.eye(3)
        P[:2, 2] = shift
        candidate = H @ P
        # Skip candidates unable to supply any currently unknown annotation.
        _, ann = warp_labels(labels, candidate, target.shape, annotation_valid)
        if not (ann & target_valid & ~result.label_valid).any():
            continue
        work = Db @ candidate @ np.linalg.inv(Da)
        initial_metrics = _score(src, dst, work, local_cfg)
        refined, ecc_ok = _refine(src, dst, work, local_cfg)
        refined_metrics = _score(src, dst, refined, local_cfg)
        # Prevent ECC from jumping to another cell or destroying the lattice.
        points = np.array([[[ss[0]*.2, ss[1]*.2], [ss[0]*.8, ss[1]*.8], [ss[0]*.5, ss[1]*.5]]], np.float32)
        displacement = np.linalg.norm(cv2.perspectiveTransform(points, work)-cv2.perspectiveTransform(points, refined), axis=2)
        limit = min(16., min(np.linalg.norm(Db[:2, :2] @ H[:2, :2] @ v) for v in vectors)*.1)
        if refined_metrics['match_score'] >= initial_metrics['match_score'] and displacement.max() <= limit:
            candidate = np.linalg.inv(Db) @ refined @ Da
            metrics = refined_metrics
        else:
            metrics, ecc_ok = initial_metrics, False
        accepted = metrics['structure_ncc'] >= cfg.periodic_min_ncc and metrics['edge_alignment'] >= cfg.periodic_min_edge
        record = dict(offset=list(offset), accepted=bool(accepted), metrics=metrics,
                      ecc_converged=bool(ecc_ok), matrix_template_to_target=candidate.tolist(), added_label_pixels=0)
        info['copies'].append(record)
        if not accepted:
            continue
        mapped, ann = warp_labels(labels, candidate, target.shape, annotation_valid)
        fill_labels = ann & target_valid & ~result.label_valid
        # Validate each new connected region, not only the large aligned interior.
        # A bounded context supplies edges for thin strips and corner fragments.
        count, components = cv2.connectedComponents(fill_labels.astype(np.uint8))
        approved = np.zeros((h, w), bool)
        record['local_regions'] = []
        candidate_work = Db @ candidate @ np.linalg.inv(Da)
        for component in range(1, count):
            region = components == component
            local = cv2.resize(region.astype(np.uint8), ts, interpolation=cv2.INTER_NEAREST)
            local = cv2.dilate(local, np.ones((25, 25), np.uint8)) & dst[3]
            local_dst = (dst[0], dst[1], dst[2], local)
            local_metrics = _score(src, local_dst, candidate_work, local_cfg)
            local_ok = (local_metrics['structure_ncc'] >= cfg.periodic_min_ncc
                        and local_metrics['edge_alignment'] >= cfg.periodic_min_edge)
            record['local_regions'].append(dict(accepted=bool(local_ok), pixels=int(region.sum()), metrics=local_metrics))
            if local_ok:
                approved |= region
        fill_labels = approved
        record['accepted'] = bool(fill_labels.any())
        if not fill_labels.any():
            continue
        result.mapped_labels[fill_labels] = mapped[fill_labels]
        result.label_valid |= fill_labels
        result.label_periodic_valid |= fill_labels
        _, geom = warp_labels(labels, candidate, target.shape, source_valid)
        fill_image = geom & target_valid & ~result.completed_valid & approved
        warped = cv2.warpPerspective(source, candidate, (w, h))
        result.completed_template[fill_image] = warped[fill_image]
        result.completed_valid |= fill_image
        result.periodic_valid |= fill_image
        record['added_label_pixels'] = int(fill_labels.sum())
        if result.label_valid[target_valid].all():
            break
    finish()
