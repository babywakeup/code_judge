"""Independent reference contours cover holes without a nearby observed seed."""

from dataclasses import replace
from itertools import product

import cv2
import numpy as np

from ...utils import ValidRegion, to_gray
from ..registration.core import _features
from ..registration.scoring import ScoreWarp, _ncc, _score
from .layers import record_periodic_prediction
from .regions import MaskRegion, RegionResize


def complete_direct_references(result, inputs, template_index, transferred, cfg, *, phase_only=False):
    """Use registered contours first, then poses verified by the shared lattice."""
    shared = cfg.periodic_completion and cfg.shared_periods
    if cfg.label_mode != 'semantic' or (phase_only and not shared):
        return
    records = []
    for index in result.registration['ranking']:
        primary = index == template_index
        if phase_only and not primary:
            continue
        if inputs.labels[index] is None:
            continue
        entry = result.registration['matches'][index]
        matrix = entry.get('matrix_template_to_target')
        if not entry.get('geometry_accepted') or matrix is None:
            continue
        source = inputs.templates[index]
        H = np.asarray(matrix, float)
        poses = [] if primary else [((1,), [], np.zeros(2), H)]
        if not phase_only and shared and entry.get('periodic_detected'):
            h, w = source.shape[:2]
            center = np.array([[[(w-1)/2, (h-1)/2]]], float)
            inverse = np.linalg.inv(H)
            for rank, candidate in enumerate(entry.get('candidates', ()), 1):
                if (candidate['match_score'] < cfg.min_score
                        or candidate['target_coverage'] < cfg.min_overlap
                        or candidate['structure_ncc'] < cfg.periodic_min_ncc
                        or candidate['edge_alignment'] < cfg.periodic_min_edge):
                    continue
                pose = np.asarray(candidate['matrix_template_to_target'], float)
                delta = (cv2.perspectiveTransform(center, inverse @ pose) - center)[0, 0]
                if np.linalg.norm(delta) >= 12:
                    poses.append(((rank,), [delta], np.zeros(2), pose))
        if phase_only:
            # The lattice proposes coordinates. Foreground still comes from
            # the source annotation and must pass its own appearance checks.
            periodic = result.registration.get('periodic_completion', {})
            axes = periodic.get('lattice', ())
            for copy in periodic.get('copies', ()):
                offset = copy['offset']
                if (not copy.get('geometry_accepted')
                        or copy.get('refinement_mode') != 'shared_lattice'
                        or copy['metrics']['target_coverage'] < cfg.min_overlap):
                    continue
                vectors = [axis['source_vector'] for n, axis in zip(offset, axes) if n]
                poses.append(((0, *offset), vectors, np.zeros(2),
                              np.asarray(copy['matrix_template_to_target'], float)))
        labels, annotation = ((transferred.labels, transferred.annotation_valid) if primary
                              else (inputs.labels[index], inputs.annotation_valids[index]))
        if not poses:
            continue
        info = dict(template_index=index, copies=[], appearance_candidates=0, added_label_pixels=0)
        _transfer_components(result, inputs.target_roi, cfg, labels, annotation,
                             inputs.template_valids[index], index, source, poses, info,
                             direct=True)
        records.append(info)
    if records:
        result.registration.setdefault('direct_completion', []).extend(records)


def complete_reference_instances(result, target_valid, cfg, *, labels,
                                 annotation_valid, source_valid, template_index,
                                 source_image=None):
    """Use a geometrically accepted source's complete, explicitly known foreground.

    An observed target contour may be two periods away while an independently
    annotated source contour is within one configured step of a registered pose.
    Direct registration poses are handled before this reference-only pass.
    """
    if not (cfg.periodic_completion and cfg.shared_periods and cfg.label_mode == 'semantic'):
        return
    info = result.registration.get('periodic_completion', {})
    lattice = info.get('lattice', [])
    if result.mapped_labels is None or result.label_valid is None:
        return
    alternate = template_index != result.registration['ranking'][0]
    if alternate:
        if source_image is None or template_index not in result.registration['ranking']:
            raise ValueError('Alternate reference contours require their ranked source image')
        candidate = result.registration['matches'][template_index]
        matrix = candidate.get('matrix_template_to_target')
        if not candidate.get('geometry_accepted') or matrix is None:
            return
        H = np.asarray(matrix, float)
    else:
        H, source_image = result.matrix_template_to_target, result.template_image
        if not lattice or H is None:
            return
    shape = source_image.shape[:2]
    if labels.shape != shape or source_valid.shape != shape or (
            annotation_valid is not None and annotation_valid.shape != shape):
        raise ValueError('Reference labels and validity must match their source image')
    offsets = []
    for offset in product(range(-cfg.periodic_max_steps, cfg.periodic_max_steps+1), repeat=len(lattice)):
        active = [axis for n, axis in zip(offset, lattice) if n]
        if not active or not all(axis.get('verified', False) for axis in active):
            continue
        vectors = [np.linalg.solve(H[:2, :2], axis['target_vector']) if alternate
                   else np.asarray(axis['source_vector'], float) for axis in active]
        shift = sum((n*np.asarray(axis['target_vector'], float)
                     for n, axis in zip(offset, lattice)), np.zeros(2))
        if np.isfinite(shift).all() and np.linalg.norm(shift) >= 1:
            C = H.copy()
            C[:2, 2] += shift
            offsets.append((offset, vectors, shift, C))
    if not offsets:
        return
    records = dict(copies=[], appearance_candidates=0, added_label_pixels=0)
    if alternate:
        records['template_index'] = template_index
        info.setdefault('alternate_reference_instances', []).append(records)
    else:
        info['reference_instances'] = records
    before = int(result.label_valid.sum()) if alternate else 0
    _transfer_components(result, target_valid, cfg, labels, annotation_valid, source_valid,
                         template_index, source_image, offsets, records, allow_unknown=alternate)
    if result.label_periodic_valid is not None:
        info['added_label_pixels'] = int(result.label_periodic_valid.sum())
    newly_covered = int(result.label_valid.sum()) - before if alternate else 0
    info['boundary_foreground_pixels'] = (info.get('boundary_foreground_pixels', 0)
                                          + records['added_label_pixels'] - newly_covered)
    if newly_covered:
        total = int(np.count_nonzero(target_valid))
        missing = int(np.count_nonzero(target_valid & ~result.label_valid))
        info.update(remaining_unknown_pixels=missing, label_coverage_roi=(total-missing)/total,
                    status='incomplete' if missing else 'complete',
                    newly_covered_pixels=info.get('newly_covered_pixels', 0)+newly_covered)


def _transfer_components(result, target_valid, cfg, labels, annotation_valid, source_valid,
                         template_index, source_image, poses, records, *, direct=False, allow_unknown=False):
    source_known = source_valid > 0
    if annotation_valid is not None:
        source_known &= annotation_valid > 0
    known = target_valid > 0
    if not (direct or allow_unknown):
        known &= result.label_valid
    occupied = (result.direct_labels if direct else result.mapped_labels) > 0
    classes = [int(c) for c in np.unique(labels[source_known])
               if c > 0 and c != cfg.ordered_object_class]
    h, w = occupied.shape
    sh, sw = labels.shape
    scale = cfg.matching_scale(source_image.shape, result.input_image.shape)
    ss = (round(sw*scale), round(sh*scale))
    ts = (round(w*scale), round(h*scale))
    Da, Db = np.diag([ss[0]/sw, ss[1]/sh, 1.]), np.diag([ts[0]/w, ts[1]/h, 1.])
    resize_region = RegionResize((h, w), ts)
    kernel = np.ones((3, 3), np.uint8)
    source_gray = target_gray = None
    proposals = []
    max_translation = min(2, max(1, round(.5/scale)))
    for class_id in classes:
        allowed = [item for item in poses
                   if (direct and not item[1]) or cfg.allows_periodic_shift(class_id, item[1])]
        if not allowed:
            continue
        blocked = ~known | occupied
        if direct:
            blocked |= (result.mapped_labels > 0) & (result.mapped_labels != class_id)
        _, components, stats, _ = cv2.connectedComponentsWithStats((labels == class_id).astype(np.uint8))
        for component, (x, y, cw, ch, pixels) in enumerate(stats[1:], 1):
            if pixels < cfg.min_contour_pixels:
                continue
            cut_edges = []
            for edge in (x, x+cw-1):
                if edge == 0 or edge == sw-1:
                    cut_edges.extend(((edge, y), (edge, y+ch-1)))
            for edge in (y, y+ch-1):
                if edge == 0 or edge == sh-1:
                    cut_edges.extend(((x, edge), (x+cw-1, edge)))
            if cut_edges and not direct:
                continue
            sx, sy = max(0, x-1), max(0, y-1)
            sr, sb = min(sw, x+cw+1), min(sh, y+ch+1)
            source_slice = np.s_[sy:sb, sx:sr]
            mask = (components[source_slice] == component).astype(np.uint8)
            halo = cv2.dilate(mask, kernel) > 0
            if (halo & (~source_known[source_slice]
                        | ((labels[source_slice] > 0) & (labels[source_slice] != class_id)))).any():
                continue
            corners = np.array([[[sx, sy], [sr, sy], [sr, sb], [sx, sb]]], np.float64)
            for offset, _, shift, C in allowed:
                if cut_edges:
                    # Preserve a real contour cut by the target crop. A source
                    # fragment projected inside the photo is incomplete evidence.
                    cut = cv2.perspectiveTransform(np.array([cut_edges], np.float64), C).reshape(-1, 2, 2)
                    margin = 1/scale
                    outside = (cut < margin).all(axis=1) | (cut >= np.array([w, h])-margin).all(axis=1)
                    if not outside.any(axis=1).all():
                        continue
                projected = cv2.perspectiveTransform(corners, C)[0]
                if not np.isfinite(projected).all():
                    continue
                xx, yy = np.floor(projected.min(axis=0)).astype(int)-1
                right, bottom = np.ceil(projected.max(axis=0)).astype(int)+1
                if direct:
                    xx, yy, right, bottom = max(0, xx), max(0, yy), min(w, right), min(h, bottom)
                if xx < 0 or yy < 0 or right > w or bottom > h or xx >= right or yy >= bottom:
                    continue
                size = (int(right-xx), int(bottom-yy))
                to_local = np.array([[1., 0., -xx], [0., 1., -yy], [0., 0., 1.]])
                from_source = np.array([[1., 0., sx], [0., 1., sy], [0., 0., 1.]])
                local = to_local @ C @ from_source
                mapped = cv2.warpPerspective(mask, local, size, flags=cv2.INTER_NEAREST) > 0
                dest_slice = np.s_[yy:bottom, xx:right]
                dest_halo = cv2.dilate(mapped.astype(np.uint8), kernel) > 0
                if not mapped.any() or (dest_halo & blocked[dest_slice]).any():
                    continue
                if source_gray is None:
                    source_gray, target_gray = to_gray(source_image), to_gray(result.input_image)
                observed = cv2.warpPerspective(source_gray[source_slice], local, size,
                                               flags=cv2.INTER_NEAREST)
                samples = observed[mapped].astype(np.float32)
                raw = _ncc(samples, target_gray[dest_slice][mapped].astype(np.float32))
                records['appearance_candidates'] += 1
                translation = (0, 0)
                if direct and not cut_edges and cfg.periodic_min_ncc-.04 <= raw < cfg.periodic_min_ncc:
                    raw, dest_slice, translation = _refine_direct_translation(
                        samples, target_gray, mapped, dest_halo, blocked, dest_slice, raw, max_translation)
                    if translation != (0, 0):
                        C = C.copy()
                        C[:2] += np.outer(translation, C[2])
                if not np.isfinite(raw) or raw < cfg.periodic_min_ncc:
                    continue
                if resize_region(MaskRegion(dest_slice[0].start, dest_slice[1].start, mapped)).sum() < 16:
                    continue
                proposals.append((raw, class_id, component, offset, shift, C, dest_slice,
                                  mapped, [int(x), int(y), int(x+cw), int(y+ch)], translation))
    if not proposals:
        return
    src = _features(*ValidRegion.resize_for_matching(source_gray, source_valid, ss), cfg=cfg)
    dst = _features(*ValidRegion.resize_for_matching(target_gray, target_valid, ts), cfg=cfg)
    local_cfg = replace(cfg, min_overlap=cfg.local_min_overlap)
    warps = {}
    for raw, class_id, component, offset, shift, C, dest_slice, mapped, source_bbox, translation in sorted(
            proposals, key=lambda proposal: -proposal[0]):
        if (mapped & occupied[dest_slice]).any():
            continue
        work = Db @ C @ np.linalg.inv(Da)
        key = offset, translation
        if key not in warps:
            warps[key] = ScoreWarp(src, work, ts)
        roi = cv2.dilate(resize_region(MaskRegion(dest_slice[0].start, dest_slice[1].start, mapped)),
                         np.ones((cfg.local_context_size,)*2, np.uint8)) & dst[3]
        metrics = _score(src, (dst[0], dst[1], dst[2], roi), work, local_cfg, warped=warps[key])
        if not (metrics['structure_ncc'] >= cfg.periodic_min_ncc
                and metrics['edge_alignment'] >= cfg.periodic_min_edge):
            continue
        result.mapped_labels[dest_slice][mapped] = class_id
        result.label_valid[dest_slice] |= mapped
        occupied[dest_slice] |= mapped
        if direct:
            result.direct_labels[dest_slice][mapped] = class_id
            result.direct_valid[dest_slice] |= mapped
            if result.label_periodic_valid is not None:
                result.label_periodic_valid[dest_slice][mapped] = False
        elif result.label_periodic_valid is not None:
            result.label_periodic_valid[dest_slice] |= mapped
            prediction = np.full(mapped.shape, class_id, dtype=result.mapped_labels.dtype)
            record_periodic_prediction(result, prediction, mapped, region=dest_slice)
        ys, xs = np.where(mapped)
        bbox = [int(dest_slice[1].start+xs.min()), int(dest_slice[0].start+ys.min()),
                int(dest_slice[1].start+xs.max()+1), int(dest_slice[0].start+ys.max()+1)]
        count = int(mapped.sum())
        record = dict(class_id=class_id, template_index=template_index,
            source_component=component, source_bbox=source_bbox, target_bbox=bbox,
            matrix_template_to_target=C.tolist(), raw_pixel_ncc=float(raw), metrics=metrics, added_label_pixels=count)
        if direct:
            record['candidate_rank'] = offset[0] or None
            if offset[0] == 0:
                record['phase_offset'] = list(offset[1:])
            if translation != (0, 0):
                record['local_translation'] = list(translation)
        else:
            record.update(offset=list(offset), target_vector=shift.tolist())
        records['copies'].append(record)
        records['added_label_pixels'] += count


def _refine_direct_translation(samples, target, mapped, halo, blocked, dest, score, radius):
    """Test a few native-pixel translations against the same source contour."""
    best = score, dest, (0, 0)
    ys, xs = dest
    height, width = target.shape
    for dy, dx in product(range(-radius, radius+1), repeat=2):
        if dx == dy == 0:
            continue
        y, x = ys.start+dy, xs.start+dx
        bottom, right = ys.stop+dy, xs.stop+dx
        if y < 0 or x < 0 or bottom > height or right > width:
            continue
        moved = np.s_[y:bottom, x:right]
        if (halo & blocked[moved]).any():
            continue
        ncc = _ncc(samples, target[moved][mapped].astype(np.float32))
        if ncc > best[0]:
            best = ncc, moved, (dx, dy)
    return best
