"""Verify support extensions perpendicular to an identified object's layout.

This never chooses new identities or repeats along the identity layout axis.
The shared canonical solver uses x layouts with y support; y layouts transpose.
Only actual source pixels with a verified transform can fill.
White defects are missing appearance evidence, not missing label knowledge:
verify on the visible portion, then transfer the entire supported source mask.
"""
from dataclasses import replace

import cv2
import numpy as np

from ...utils import ValidRegion, to_gray, warp_valid_mask
from ..segmentation.periodic import _candidate_fill
from ..registration.core import _features
from ..registration.scoring import _ncc, _score
from .axes import transpose_image, transpose_config, transpose_prediction


def _white_occlusions(gray, valid):
    """Large near-white patches are excluded from evidence, never from output ROI.

    Keep isolated highlights/noise as evidence. Connected area and brightness
    describe the white-defect prior, not a license to discard arbitrary mismatch.
    Acceptance still needs contrast and at least half the actual fill pixels.
    """
    white = ((gray >= 230) & (valid > 0)).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(white)
    large = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= 64)
    return np.isin(components, large[large != 0])


def complete_object_support(prediction, images, references, source_valids,
                            target, target_valid, target_vectors, cfg):
    if cfg.object_order_axis == 'y':
        canonical = transpose_prediction(prediction)
        complete_object_support(canonical, [transpose_image(im) for im in images],
            [[dict(r, mask=transpose_image(r['mask'])) for r in refs] for refs in references],
            [transpose_image(v) for v in source_valids], transpose_image(target), transpose_image(target_valid),
            [[v[1], v[0]] for v in target_vectors], transpose_config(cfg))
        restored = transpose_prediction(canonical)
        prediction.labels[:] = restored.labels
        prediction.identities[:] = restored.identities
        prediction.valid[:] = restored.valid
        # Preserve the registration dictionary's reference to this diagnostics map.
        prediction.diagnostics.clear()
        prediction.diagnostics.update(restored.diagnostics)
        return
    if prediction.diagnostics.get('status') not in ('accepted', 'partial_reference'):
        return
    vectors = [np.asarray(v, float) for v in target_vectors
               if abs(v[1]) >= 12 and abs(v[0]) <= .15*abs(v[1])
               and cfg.allows_periodic_shift(cfg.ordered_object_class, [v])]
    info = dict(added_valid_pixels=0, added_foreground_pixels=0, copies=[],
                repeat_axes=list(cfg.periodic_axes(cfg.ordered_object_class)))
    prediction.diagnostics['support_completion'] = info
    if not vectors or prediction.valid[target_valid > 0].all():
        return
    shape = target.shape[:2]
    scale = cfg.matching_scale(target.shape, *(im.shape for im in images))
    ts = (round(shape[1]*scale), round(shape[0]*scale))
    Db = np.diag([ts[0]/shape[1], ts[1]/shape[0], 1.])
    target_gray = to_gray(target)
    occluded = _white_occlusions(target_gray, target_valid)
    info['occluded_pixels'] = int(occluded.sum())
    evidence_valid = ((target_valid > 0) & ~occluded).astype(np.uint8)
    if not evidence_valid.any():
        return
    tg, tm = ValidRegion.resize_for_matching(target_gray, evidence_valid, ts)
    dst = _features(tg, tm)
    cache = {}
    local_cfg = replace(cfg, min_overlap=.2)
    for obj in prediction.diagnostics.get('objects', []):
        ti, ri = obj['template_index'], obj['reference_index']
        # A vertically shifted fragment cannot establish unseen horizontal extent.
        if obj.get('reference_partial', False):
            continue
        ref = references[ti][ri]
        raw = ((ref['mask'] > 0) & (source_valids[ti] > 0)).astype(np.uint8)
        if ti not in cache:
            ss = (round(images[ti].shape[1]*scale), round(images[ti].shape[0]*scale))
            Da = np.diag([ss[0]/images[ti].shape[1], ss[1]/images[ti].shape[0], 1.])
            sg, sm = ValidRegion.resize_for_matching(to_gray(images[ti]), source_valids[ti], ss)
            cache[ti] = _features(sg, sm), Da
        src, Da = cache[ti]
        H = np.asarray(obj['matrix_template_to_target'])
        identity = list(cfg.object_cycle).index(obj['identity'])+1
        for vector in vectors:
            for step in [s for n in range(1, cfg.periodic_max_steps+1) for s in (n, -n)]:
                shift = np.eye(3)
                shift[:2, 2] = step*vector
                candidate = shift @ H
                mapped = warp_valid_mask(raw, candidate, target.shape).astype(np.uint16)*cfg.ordered_object_class
                own = (prediction.identities == identity) | (mapped > 0)
                _, xs = np.where(own)
                if not len(xs):
                    continue
                band = np.zeros(shape, bool)
                band[:, xs.min():xs.max()+1] = True
                annotation = warp_valid_mask(source_valids[ti], candidate, target.shape) & band
                if not ref.get('background_known', True):
                    annotation &= mapped > 0
                if not (annotation & (target_valid > 0) & ~prediction.valid).any():
                    continue
                fill = _candidate_fill(mapped, annotation, target_valid, prediction.labels, prediction.valid)
                fill &= (prediction.labels == 0)
                count, components = cv2.connectedComponents(fill.astype(np.uint8))
                observed_source = cv2.warpPerspective(to_gray(images[ti]), candidate, (shape[1], shape[0]))
                work = Db @ candidate @ np.linalg.inv(Da)
                work_source = cv2.warpPerspective(src[0], work, ts)
                work_support = cv2.warpPerspective(src[3], work, ts, flags=cv2.INTER_NEAREST)
                record = dict(identity=obj['identity'], step=step, target_vector=vector.tolist(),
                              matrix_template_to_target=candidate.tolist(), regions=[])
                approved = np.zeros(shape, bool)
                approved_labels = np.zeros(shape, np.uint16)
                pending = [(components == component, False) for component in range(1, count)]
                tile_size = max(16, round(32/scale))
                while pending:
                    region, is_tile = pending.pop()
                    local = cv2.resize(region.astype(np.uint8), ts, interpolation=cv2.INTER_NEAREST)
                    local = cv2.dilate(local, np.ones((25, 25), np.uint8)) & dst[3]
                    metrics = _score(src, (dst[0], dst[1], dst[2], local),
                                     Db @ candidate @ np.linalg.inv(Da), local_cfg)
                    # Check original-resolution pixels INSIDE the proposed fill,
                    # but do not ask a white defect to resemble healthy texture.
                    # Enough visible pixels must verify this SAME transform;
                    # fully unsupported or arbitrary mismatching regions fail.
                    evidence = region & ~occluded
                    a = observed_source[evidence].astype(np.float32)
                    b = target_gray[evidence].astype(np.float32)
                    evidence_ok = (len(a) >= max(64, .5*int(region.sum()))
                                   and np.std(a) >= 2 and np.std(b) >= 2)
                    region_ncc = _ncc(a, b) if evidence_ok else 0.
                    accepted = (metrics['structure_ncc'] >= cfg.periodic_min_ncc
                                and metrics['edge_alignment'] >= cfg.periodic_min_edge
                                and region_ncc >= cfg.periodic_min_ncc)
                    region_mapped = mapped
                    corrected = None
                    # A lattice supplies an initial displacement, not exact local
                    # pixel alignment. Refine translation only, bounded far below
                    # one period, and recheck the actual new pixels afterwards.
                    ecc_mask = local & work_support
                    if not accepted and evidence_ok and ecc_mask.sum() >= 1000:
                        W = np.eye(2, 3, dtype=np.float32)
                        try:
                            _, W = cv2.findTransformECC(dst[0], work_source, W, cv2.MOTION_TRANSLATION,
                                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, cfg.ecc_iterations, 1e-5),
                                ecc_mask*255, 5)
                            if np.linalg.norm(W[:, 2]) <= 6.:
                                refined = np.linalg.inv(Db) @ np.linalg.inv(np.vstack([W, [0, 0, 1]])) @ Db @ candidate
                                support = warp_valid_mask(source_valids[ti], refined, target.shape)
                                if support[region].all():
                                    observed = cv2.warpPerspective(to_gray(images[ti]), refined, (shape[1], shape[0]))
                                    values = observed[evidence].astype(np.float32)
                                    refined_ncc = _ncc(values, b) if np.std(values) >= 2 else 0.
                                    refined_metrics = _score(src, (dst[0], dst[1], dst[2], local),
                                                            Db @ refined @ np.linalg.inv(Da), local_cfg)
                                    if (refined_ncc >= cfg.periodic_min_ncc
                                            and refined_metrics['structure_ncc'] >= cfg.periodic_min_ncc
                                            and refined_metrics['edge_alignment'] >= cfg.periodic_min_edge):
                                        # Paste the mask using the transform that
                                        # actually passed; never score a correction
                                        # then paste labels with the old transform.
                                        region_mapped = warp_valid_mask(raw, refined, target.shape).astype(np.uint16)*cfg.ordered_object_class
                                        if not ref.get('background_known', True) and not (region_mapped[region] > 0).all():
                                            continue
                                        accepted, corrected = True, refined.tolist()
                                        metrics, region_ncc = refined_metrics, refined_ncc
                        except (cv2.error, np.linalg.LinAlgError):
                            pass
                    record['regions'].append(dict(accepted=bool(accepted), pixels=int(region.sum()),
                                                  evidence_pixels=int(evidence.sum()),
                                                  occluded_pixels=int((region & occluded).sum()),
                                                  region_ncc=region_ncc, local_tile=is_tile, metrics=metrics,
                                                  corrected_matrix=corrected))
                    if accepted:
                        approved |= region
                        approved_labels[region] = region_mapped[region]
                        if corrected is not None:
                            obj.setdefault('supporting_transforms', []).append(corrected)
                    elif not is_tile:
                        # One occluded patch must not reject an entire connected
                        # vertical column. Retry bounded local pieces, each with
                        # the SAME direct-pixel and context evidence requirements.
                        yy, xx = np.where(region)
                        if max(yy.max()-yy.min()+1, xx.max()-xx.min()+1) > tile_size:
                            for y in range(int(yy.min()), int(yy.max())+1, tile_size):
                                for x in range(int(xx.min()), int(xx.max())+1, tile_size):
                                    piece = region[y:y+tile_size, x:x+tile_size]
                                    if not piece.any():
                                        continue
                                    tile = np.zeros(shape, bool)
                                    tile[y:y+tile_size, x:x+tile_size] = piece
                                    pending.append((tile, True))
                record['accepted'] = bool(approved.any())
                info['copies'].append(record)
                if not approved.any():
                    continue
                positive = approved & (approved_labels > 0)
                info['added_valid_pixels'] += int((approved & ~prediction.valid).sum())
                info['added_foreground_pixels'] += int(positive.sum())
                prediction.labels[approved] = approved_labels[approved]
                prediction.identities[positive] = identity
                prediction.valid |= approved
                obj.setdefault('supporting_transforms', []).append(candidate.tolist())
    prediction.diagnostics['valid_coverage_roi'] = float(
        (prediction.valid & (target_valid > 0)).sum()/max(int((target_valid > 0).sum()), 1))
