"""One verified translation lattice shared by base labels and object support.

Axis permissions always refer to SOURCE coordinates. Target vectors may be
skewed. A periodic identity window is not a license to tile different identities.
Only geometry is shared; every newly filled region still supplies local evidence.
"""
from collections import defaultdict
from dataclasses import replace

import cv2
import numpy as np

from ..registration.scoring import ScoreWarp, _refine, _score, prepare_refinement_target


def _translation(vector):
    matrix = np.eye(3)
    matrix[:2, 2] = vector
    return matrix


def _accepted(metrics, cfg):
    return (metrics['structure_ncc'] >= cfg.periodic_min_ncc
            and metrics['edge_alignment'] >= cfg.periodic_min_edge)


def _fit_translation(features, vector, to_work, cfg, prepared, *, fixed=False):
    """Calibrate one fundamental period, never independently deform its copies."""
    work = _translation(to_work[:2, :2] @ vector)
    size = (features[0].shape[1], features[0].shape[0])
    proposal = ScoreWarp(features, work, size)
    before = _score(features, features, work, cfg, warped=proposal)
    selected, metrics, converged = work, before, False
    if not fixed and not (before['structure_ncc'] >= 1-1e-6 and before['edge_alignment'] >= 1-1e-6):
        refined, ok = _refine(features, features, work, cfg, prepared=prepared,
                              initial_warp=proposal, motion=cv2.MOTION_TRANSLATION)
        limit = min(6., .1*np.linalg.norm(work[:2, 2]))
        if ok and np.linalg.norm(refined[:2, 2]-work[:2, 2]) <= limit:
            after = _score(features, features, refined, cfg)
            if after['match_score'] >= before['match_score'] and _accepted(after, cfg):
                selected, metrics, converged = refined, after, True
    fitted = np.linalg.solve(to_work[:2, :2], selected[:2, 2])
    return fitted, dict(verified=bool(_accepted(metrics, cfg)), ecc_converged=converged, metrics=metrics)


def _occurrence_hints(objects, nominal, to_work, cfg):
    """Same reference/identity aliases may propose, but cannot certify, a period.

    Fold integer multiples before robust aggregation. Never use horizontal
    A/B/C spacing as evidence for one identity's vertical repeat distance.
    """
    candidates = []
    for obj in objects.get('objects', []):
        groups = defaultdict(list)
        for ref in obj.get('supporting_references', []):
            groups[(ref['template_index'], ref['reference_index'])].append(
                np.asarray(ref['matrix_template_to_target']))
        for matrices in groups.values():
            for i, a in enumerate(matrices):
                for b in matrices[i+1:]:
                    if np.max(np.abs(a[:2, :2]-b[:2, :2])) > .02:
                        continue
                    delta = b[:2, 2]-a[:2, 2]
                    multiple = round(float(np.dot(delta, nominal)/max(np.dot(nominal, nominal), 1e-9)))
                    if multiple == 0 or abs(multiple) > cfg.periodic_max_steps+1:
                        continue
                    proposal = delta/multiple
                    if np.linalg.norm(to_work[:2, :2] @ (proposal-nominal)) <= 6:
                        candidates.append(proposal)
    return np.median(candidates, axis=0) if candidates else nominal, len(candidates)


def calibrate_lattice(src, dst, vectors, matrix, source_to_work, target_to_work, cfg, *,
                      configured=False, objects=None):
    """Joint source-image, target-image and same-identity occurrence evidence.

    Configured source vectors are kept fixed. Target image verification remains
    mandatory, including after object aliases propose a refined displacement.
    The context is serializable and is reused without re-estimating each copy.
    """
    local_cfg = replace(cfg, min_overlap=.2)
    source_levels = prepare_refinement_target(src)
    target_levels = prepare_refinement_target(dst)
    records = []
    for vector in vectors:
        original = np.asarray(vector, float)
        source, source_info = _fit_translation(src, original, source_to_work, local_cfg,
                                                source_levels, fixed=configured)
        nominal = matrix[:2, :2] @ source
        target, hints = _occurrence_hints(objects or {}, nominal, target_to_work, cfg)
        if source_info['verified']:
            target, target_info = _fit_translation(dst, target, target_to_work, local_cfg, target_levels)
        else:
            source, target = original, matrix[:2, :2] @ original
            target_info = dict(verified=False, ecc_converged=False, metrics=None)
        verified = bool(source_info['verified'] and target_info['verified'])
        if not verified:
            # Keep the original prior for the legacy per-copy verification path.
            source, target = original, matrix[:2, :2] @ original
        records.append(dict(source_vector=source.tolist(), target_vector=target.tolist(),
                            initial_source_vector=original.tolist(), verified=verified,
                            occurrence_hint_count=hints,
                            source_verification=source_info, target_verification=target_info))
    return records