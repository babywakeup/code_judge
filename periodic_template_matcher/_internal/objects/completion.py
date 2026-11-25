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
from ..segmentation.regions import MaskRegion, RegionResize, component_regions
from ..registration.core import _features
from ..registration.scoring import ScoreWarp, _ncc, _score, _ecc_translation
from .axes import transpose_image, transpose_config, transpose_prediction
from .support_anchors import (canonical_support_objects, corroborate_clipped_bridges,
                              interior_bridges, verified_repeat_regions)


def _white_occlusions(gray, valid, cfg):
    """Large near-white patches are excluded from evidence, never from output ROI.

    Keep isolated highlights/noise as evidence. Connected area and brightness
    describe the white-defect prior, not a license to discard arbitrary mismatch.
    Acceptance still needs contrast and at least half the actual fill pixels.
    """
    white = ((gray >= cfg.object_white_level) & (valid > 0)).astype(np.uint8)
    _, components, stats, _ = cv2.connectedComponentsWithStats(white)
    large = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= cfg.object_white_min_area)
    return np.isin(components, large[large != 0])


def support_vectors(target_vectors, cfg, *, source_vectors=None):
    """Select verified shifts perpendicular to the object order axis."""
    if source_vectors is not None:
        if len(source_vectors) != len(target_vectors):
            raise ValueError('Each target period requires its source-axis vector')
        return [np.asarray(target, float) for source, target in zip(source_vectors, target_vectors)
                if np.linalg.norm(target) >= 12
                and cfg.allows_periodic_shift(cfg.ordered_object_class, [source])]
    along = 0 if cfg.object_order_axis == 'x' else 1
    across = 1 - along
    return [np.asarray(v, float) for v in target_vectors
            if abs(v[across]) >= 12 and abs(v[along]) <= .15*abs(v[across])
            and cfg.allows_periodic_shift(cfg.ordered_object_class, [v])]


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
    lattice = prediction.diagnostics.get('periodic_lattice', []) if cfg.shared_periods else []
    sources = ([axis['source_vector'] for axis in lattice]
               if len(lattice) == len(target_vectors) and lattice else None)
    vectors = support_vectors(target_vectors, cfg, source_vectors=sources)
    info = dict(added_valid_pixels=0, added_foreground_pixels=0, copies=[],
                repeat_axes=list(cfg.periodic_axes(cfg.ordered_object_class)),
                bridged_foreground_pixels=0, verified_repeat_foreground_pixels=0,
                clipped_bridge_pixels=0, clipped_bridges=[])
    prediction.diagnostics['support_completion'] = info
    if not vectors or prediction.valid[target_valid > 0].all():
        return
    shape = target.shape[:2]
    scale = cfg.matching_scale(target.shape, *(im.shape for im in images))
    ts = (round(shape[1]*scale), round(shape[0]*scale))
    resize_region = RegionResize(shape, ts)
    Db = np.diag([ts[0]/shape[1], ts[1]/shape[0], 1.])
    target_gray = to_gray(target)
    occluded = _white_occlusions(target_gray, target_valid, cfg)
    info['occluded_pixels'] = int(occluded.sum())
    evidence_valid = ((target_valid > 0) & ~occluded).astype(np.uint8)
    if not evidence_valid.any():
        return
    tg, tm = ValidRegion.resize_for_matching(target_gray, evidence_valid, ts)
    dst = _features(tg, tm, cfg=cfg)
    cache = {}
    gray_cache = {}
    local_cfg = replace(cfg, min_overlap=cfg.local_min_overlap)
    target_roi = target_valid > 0
    # One residual translation per identity/reference and periodic direction;
    # a failed solve is also cached so fragmented regions cannot trigger it N times.
    translation_cache = {}
    if cfg.shared_periods:
        info.update(shared_translation_fits=0, shared_translation_reuses=0,
                    source_period_vectors=sources)
    confirmed = None
    projections = {}
    attempted = set()
    repeat_warps = {}

    def support_objects():
        # First use the existing cheap/verified copies. Most images become
        # complete here and should pay no canonical-ranking/projection cost.
        yield from prediction.diagnostics.get('objects', [])
        info['canonical_anchors'] = []
        if prediction.valid[target_roi].all():
            return
        canonical = canonical_support_objects(prediction, images, references, source_valids,
            vectors, target_gray, ts, cfg, target_valid=evidence_valid)
        info['canonical_anchors'] = [dict(identity=o['identity'], template_index=o['template_index'],
            reference_index=o['reference_index'], matrix_template_to_target=o['matrix_template_to_target'],
            **o['support_anchor']) for o in canonical]
        yield from canonical

    # Preserve the old accepted positives before adding alternative coverage.
    for obj in support_objects():
        ti, ri = obj['template_index'], obj['reference_index']
        # A vertically shifted fragment cannot establish unseen horizontal extent.
        if obj.get('reference_partial', False):
            continue
        ref = references[ti][ri]
        raw = ((ref['mask'] > 0) & (source_valids[ti] > 0)).astype(np.uint8)
        if ti not in cache:
            ss = (round(images[ti].shape[1]*scale), round(images[ti].shape[0]*scale))
            Da = np.diag([ss[0]/images[ti].shape[1], ss[1]/images[ti].shape[0], 1.])
            gray_cache[ti] = to_gray(images[ti])
            sg, sm = ValidRegion.resize_for_matching(gray_cache[ti], source_valids[ti], ss)
            cache[ti] = _features(sg, sm, cfg=cfg), Da
        src, Da = cache[ti]
        H = np.asarray(obj['matrix_template_to_target'])
        identity = list(cfg.object_cycle).index(obj['identity'])+1
        for vector in vectors:
            anchor = obj.get('support_anchor')
            if anchor and not np.allclose(vector, anchor['target_vector']):
                continue
            if anchor and confirmed is None:
                # Legacy fills passed their own direct appearance checks. Freeze
                # them too, BEFORE any bridge can contribute anchor evidence.
                confirmed = replace(prediction, identities=prediction.identities.copy(),
                                    valid=prediction.valid.copy())
            if anchor:
                steps = [0, -1, 1] if cfg.periodic_max_steps else [0]
            else:
                steps = [s for n in range(1, cfg.periodic_max_steps+1) for s in (n, -n)]
            bridge_only = bool(anchor and anchor.get('bridge_only'))
            clipped = []
            bridged = None
            for step in steps:
                shift = np.eye(3)
                shift[:2, 2] = step*vector
                candidate = shift @ H
                mapped = warp_valid_mask(raw, candidate, target.shape).astype(np.uint16)*cfg.ordered_object_class
                own = (prediction.identities == identity) | (mapped > 0)
                xs = np.flatnonzero(own.any(axis=0))
                if not len(xs):
                    continue
                band = np.s_[:, xs[0]:xs[-1]+1]
                local_candidate = candidate.copy()
                local_candidate[0] -= xs[0] * candidate[2]
                annotation = warp_valid_mask(source_valids[ti], local_candidate, mapped[band].shape)
                # Extra canonical coverage addresses the observed object, not
                # surrounding background. Keep unverified background unknown
                # and avoid rescoring large irrelevant background rectangles.
                if anchor or not ref.get('background_known', True):
                    annotation &= mapped[band] > 0
                missing = (prediction.labels[band] == 0 if bridge_only else ~prediction.valid[band])
                missing &= annotation & target_roi[band]
                if not missing.any() or (anchor and step and np.count_nonzero(missing) < cfg.min_contour_pixels):
                    continue
                fill = np.zeros(shape, bool)
                if not (bridge_only or (anchor and step)):
                    fill[band] = _candidate_fill(mapped[band], annotation, target_valid[band],
                                                 prediction.labels[band], prediction.valid[band])
                    fill[band] &= prediction.labels[band] == 0
                key = ti, ri, tuple(np.round(candidate.ravel(), 6))
                if key in projections:
                    observed_source, work, work_source, work_support, score_warp = projections[key]
                else:
                    observed_source = cv2.warpPerspective(gray_cache[ti], candidate, (shape[1], shape[0]))
                    work = Db @ candidate @ np.linalg.inv(Da)
                    work_source = cv2.warpPerspective(src[0], work, ts)
                    work_support = cv2.warpPerspective(src[3], work, ts, flags=cv2.INTER_NEAREST)
                    score_warp = ScoreWarp(src, work, ts, intensity=work_source, support=work_support)
                    # Bounded one-pose cache: fragmented duplicates reuse their
                    # projections without retaining N original-sized images.
                    projections.clear()
                    projections[key] = observed_source, work, work_source, work_support, score_warp
                record = dict(identity=obj['identity'], step=step, target_vector=vector.tolist(),
                              template_index=ti, reference_index=ri,
                              matrix_template_to_target=candidate.tolist(), regions=[])
                approved = np.zeros(shape, bool)
                approved_labels = np.zeros(shape, np.uint16)
                repeat_fill = None
                if anchor:
                    def verify_end(mask, pixel_ncc):
                        if pixel_ncc < cfg.periodic_min_ncc:
                            return None
                        local = resize_region(mask)
                        # Context may extend ACROSS the contour, not into the
                        # missing interior whose appearance is not an anchor.
                        local = cv2.dilate(local, np.ones((1, 25), np.uint8)) & dst[3]
                        metrics = _score(src, (dst[0], dst[1], dst[2], local),
                                         work, local_cfg, warped=score_warp)
                        if (metrics['structure_ncc'] < cfg.periodic_min_ncc
                                or metrics['edge_alignment'] < cfg.periodic_min_edge):
                            return None
                        return dict(metrics, region_ncc=pixel_ncc)

                    # Retry one verified phase in either direction: moving the
                    # source window can expose a real anchor beyond a defect.
                    height = max(16, round((12 if step else 32)/scale))
                    bridge, ends = interior_bridges(mapped, confirmed, identity, observed_source,
                        target_gray, occluded, np.linalg.norm(vector)*cfg.periodic_max_steps,
                        height, verify_end, phase_retry=bool(step),
                        include_background=bridge_only, clipped_candidates=clipped, columns=band[1])
                    bridge &= target_roi & (prediction.labels == 0)
                    if not bridge_only:
                        bridge &= ~prediction.valid
                    pixels = int(bridge.sum())
                    if pixels:
                        if bridged is None:
                            bridged = bridge
                        else:
                            bridged |= bridge
                    approved |= bridge
                    approved_labels[bridge] = mapped[bridge]
                    fill &= ~bridge
                    record.update(canonical_anchor=anchor, interior_bridges=ends)
                    info['bridged_foreground_pixels'] += pixels
                    if not bridge_only and not step and cfg.periodic_max_steps:
                        repeat_fill = fill & ~prediction.valid
                # Canonical rephasing may select an already tried pose. Its
                # bridge checks are new, but direct scoring/ECC need not repeat.
                if (anchor and step) or key in attempted:
                    fill[:] = False
                attempted.add(key)
                regions = [region for region in component_regions(fill) if region.pixels >= cfg.min_contour_pixels]
                pending = [(region, False) for region in regions]
                tile_size = max(16, round(32/scale))
                group_key = (obj['identity'], ti, ri, tuple(vector))
                corrected_projection = None
                while pending:
                    region, is_tile = pending.pop()
                    slices, pixels = region.slices, region.pixels
                    local = resize_region(region)
                    local = cv2.dilate(local, np.ones((cfg.local_context_size,)*2, np.uint8)) & dst[3]
                    # Check original-resolution pixels INSIDE the proposed fill,
                    # but do not ask a white defect to resemble healthy texture.
                    # Enough visible pixels must verify this SAME transform;
                    # fully unsupported or arbitrary mismatching regions fail.
                    evidence = region.mask & ~occluded[slices]
                    a = observed_source[slices][evidence].astype(np.float32)
                    b = target_gray[slices][evidence].astype(np.float32)
                    evidence_ok = (len(a) >= max(64, .5*pixels)
                                   and np.std(a) >= 2 and np.std(b) >= 2)
                    region_ncc = _ncc(a, b) if evidence_ok else 0.
                    # Raw evidence is necessary for acceptance. Avoid expensive
                    # context scores when it fails; bounded rescue still runs.
                    if region_ncc >= cfg.periodic_min_ncc:
                        metrics = _score(src, (dst[0], dst[1], dst[2], local),
                                         work, local_cfg, warped=score_warp)
                    else:
                        metrics = dict(match_score=0., structure_ncc=0., edge_alignment=0.,
                                       local_consistency=0., target_coverage=0.,
                                       edge_distance_p95_work_px=None, evaluated=False)
                    accepted = (metrics['structure_ncc'] >= cfg.periodic_min_ncc
                                and metrics['edge_alignment'] >= cfg.periodic_min_edge
                                and region_ncc >= cfg.periodic_min_ncc)
                    region_mapped = mapped
                    corrected = None
                    # A lattice supplies an initial displacement, not exact local
                    # pixel alignment. Refine translation only, bounded far below
                    # one period, and recheck the actual new pixels afterwards.
                    ecc_mask = local & work_support if not accepted and evidence_ok else None
                    if ecc_mask is not None and ecc_mask.sum() >= 1000:
                        try:
                            if cfg.shared_periods and group_key in translation_cache:
                                W = translation_cache[group_key]
                                info['shared_translation_reuses'] += 1
                                if W is None:
                                    raise cv2.error('No verified group translation')
                            elif cfg.shared_periods:
                                translation_cache[group_key] = None
                                # Fit over the union of visible support, not a
                                # single failed tile; each tile is rechecked below.
                                union = work_support & dst[3]
                                context = cv2.dilate(cv2.resize((mapped > 0).astype(np.uint8), ts,
                                    interpolation=cv2.INTER_NEAREST), np.ones((cfg.local_context_size,)*2, np.uint8))
                                union &= context
                                if union.sum() < 1000:
                                    raise cv2.error('Insufficient group support')
                                info['shared_translation_fits'] += 1
                                W = _ecc_translation(dst[0], work_source, union, cfg)
                                translation_cache[group_key] = W if np.linalg.norm(W[:, 2]) <= 6. else None
                            else:
                                W = np.eye(2, 3, dtype=np.float32)
                                _, W = cv2.findTransformECC(dst[0], work_source, W, cv2.MOTION_TRANSLATION,
                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, cfg.ecc_iterations, 1e-5),
                                    ecc_mask*255, 5)
                            if np.linalg.norm(W[:, 2]) <= 6.:
                                refined = np.linalg.inv(Db) @ np.linalg.inv(np.vstack([W, [0, 0, 1]])) @ Db @ candidate
                                if corrected_projection is None or not cfg.shared_periods:
                                    support = warp_valid_mask(source_valids[ti], refined, target.shape)
                                    observed = cv2.warpPerspective(gray_cache[ti], refined, (shape[1], shape[0]))
                                    refined_work = Db @ refined @ np.linalg.inv(Da)
                                    refined_warp = ScoreWarp(src, refined_work, ts)
                                    region_labels = warp_valid_mask(raw, refined, target.shape).astype(np.uint16)*cfg.ordered_object_class
                                    corrected_projection = support, observed, refined_work, refined_warp, region_labels
                                support, observed, refined_work, refined_warp, region_labels = corrected_projection
                                if support[slices][region.mask].all():
                                    values = observed[slices][evidence].astype(np.float32)
                                    refined_ncc = _ncc(values, b) if np.std(values) >= 2 else 0.
                                    if refined_ncc >= cfg.periodic_min_ncc:
                                        refined_metrics = _score(src, (*dst[:3], local), refined_work,
                                                                 local_cfg, warped=refined_warp)
                                    else:
                                        refined_metrics = None
                                    if (refined_metrics is not None
                                            and refined_metrics['structure_ncc'] >= cfg.periodic_min_ncc
                                            and refined_metrics['edge_alignment'] >= cfg.periodic_min_edge):
                                        # Paste the mask using the transform that
                                        # actually passed; never score a correction
                                        # then paste labels with the old transform.
                                        region_mapped = region_labels
                                        if not ref.get('background_known', True) and not (region_mapped[slices][region.mask] > 0).all():
                                            continue
                                        accepted, corrected = True, refined.tolist()
                                        metrics, region_ncc = refined_metrics, refined_ncc
                        except (cv2.error, np.linalg.LinAlgError):
                            pass
                    record['regions'].append(dict(accepted=bool(accepted), pixels=pixels,
                                                  target_bbox=[region.x, region.y,
                                                      region.x+region.mask.shape[1], region.y+region.mask.shape[0]],
                                                  evidence_pixels=len(a),
                                                  occluded_pixels=pixels-len(a),
                                                  region_ncc=region_ncc, local_tile=is_tile, metrics=metrics,
                                                  corrected_matrix=corrected))
                    if accepted:
                        approved[slices] |= region.mask
                        approved_labels[slices][region.mask] = region_mapped[slices][region.mask]
                        if corrected is not None:
                            obj.setdefault('supporting_transforms', []).append(corrected)
                    elif not is_tile:
                        # One occluded patch must not reject an entire connected
                        # vertical column. Retry bounded local pieces, each with
                        # the SAME direct-pixel and context evidence requirements.
                        yy, xx = np.where(region.mask)
                        if max(yy.max()-yy.min()+1, xx.max()-xx.min()+1) > tile_size:
                            for y in range(int(yy.min()), int(yy.max())+1, tile_size):
                                for x in range(int(xx.min()), int(xx.max())+1, tile_size):
                                    piece = region.mask[y:y+tile_size, x:x+tile_size]
                                    if piece.sum() < cfg.min_contour_pixels:
                                        continue
                                    pending.append((MaskRegion(region.y+y, region.x+x, piece), True))
                if repeat_fill is not None:
                    # Give verified template support first choice. Splitting a
                    # known region before scoring can lose its original evidence.
                    repeat_fill &= ~approved
                    if np.count_nonzero(repeat_fill) >= cfg.min_contour_pixels:
                        def verify_repeat(region, pixel_ncc, direction):
                            cache_key = tuple(direction * vector)
                            if cache_key not in repeat_warps:
                                transform = np.eye(3)
                                transform[:2, 2] = Db[:2, :2] @ (direction * vector)
                                repeat_warps[cache_key] = ScoreWarp(dst, transform, ts)
                            warped = repeat_warps[cache_key]
                            local = cv2.dilate(resize_region(region), np.ones((cfg.local_context_size,)*2, np.uint8)) & dst[3]
                            metrics = _score(dst, (*dst[:3], local), warped.H, local_cfg, warped=warped)
                            if (metrics['structure_ncc'] < cfg.periodic_min_ncc
                                    or metrics['edge_alignment'] < cfg.periodic_min_edge):
                                return None
                            return dict(metrics, region_ncc=pixel_ncc)

                        repeat, repeats = verified_repeat_regions(repeat_fill, mapped, confirmed, identity,
                            target_gray, occluded, vector, tile_size, cfg.periodic_min_ncc, verify_repeat)
                        approved |= repeat
                        approved_labels[repeat] = mapped[repeat]
                        record['verified_repeats'] = repeats
                        info['verified_repeat_foreground_pixels'] += int(repeat.sum())
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
            if clipped and bridged is not None:
                for region, direction, endpoint in corroborate_clipped_bridges(
                        clipped, bridged, confirmed, identity, vector, max(1, round(1/scale))):
                    slices = region.slices
                    fill = region.mask & target_roi[slices] & (prediction.labels[slices] == 0)
                    pixels = int(fill.sum())
                    if not pixels:
                        continue
                    info['clipped_bridge_pixels'] += pixels
                    info['added_valid_pixels'] += int((fill & ~prediction.valid[slices]).sum())
                    info['added_foreground_pixels'] += pixels
                    info['clipped_bridges'].append(dict(identity=obj['identity'], step=direction,
                        target_bbox=[region.x, region.y, slices[1].stop, slices[0].stop],
                        pixels=pixels, anchor=endpoint))
                    prediction.labels[slices][fill] = cfg.ordered_object_class
                    prediction.identities[slices][fill] = identity
                    prediction.valid[slices] |= fill
    prediction.diagnostics['valid_coverage_roi'] = float(
        (prediction.valid & (target_valid > 0)).sum()/max(int((target_valid > 0).sum()), 1))
