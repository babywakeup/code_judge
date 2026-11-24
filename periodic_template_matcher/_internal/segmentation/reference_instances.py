"""Independent reference contours cover holes without a nearby observed seed."""

from dataclasses import replace
from itertools import product

import cv2
import numpy as np

from ...utils import ValidRegion, to_gray
from ..registration.core import _features
from ..registration.scoring import ScoreWarp, _ncc, _score
from .regions import MaskRegion, RegionResize


def complete_reference_instances(result, target_valid, cfg, *, labels,
                                 annotation_valid, source_valid, template_index,
                                 source_image=None):
    """Use a geometrically accepted source's complete, explicitly known foreground.

    An observed target contour may be two periods away while an independently
    annotated source contour is within one configured step of a registered pose.
    An alternate registration can also provide a missing contour at its direct pose.
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
    if alternate:
        offsets.append(((0,) * len(lattice), [], np.zeros(2), H))
    for offset in product(range(-cfg.periodic_max_steps, cfg.periodic_max_steps+1), repeat=len(lattice)):
        active = [axis for n, axis in zip(offset, lattice) if n]
        if not active or not all(axis.get('verified', False) for axis in active):
            continue
        vectors = [np.asarray(axis['source_vector'], float) for axis in active]
        shift = sum((n*np.asarray(axis['target_vector'], float)
                     for n, axis in zip(offset, lattice)), np.zeros(2))
        if np.isfinite(shift).all() and np.linalg.norm(shift) >= 1:
            C = H.copy()
            C[:2, 2] += shift
            offsets.append((offset, vectors, shift, C))
    if not offsets:
        return
    source_known = source_valid > 0
    if annotation_valid is not None:
        source_known &= annotation_valid > 0
    known = result.label_valid & (target_valid > 0)
    occupied = result.mapped_labels > 0
    classes = [int(c) for c in np.unique(labels[source_known])
               if c > 0 and c != cfg.ordered_object_class and cfg.periodic_axes(int(c))]
    records = dict(copies=[], appearance_candidates=0, added_label_pixels=0)
    if alternate:
        records['template_index'] = template_index
        info.setdefault('alternate_reference_instances', []).append(records)
    else:
        info['reference_instances'] = records
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
    for class_id in classes:
        allowed = [item for item in offsets if cfg.allows_periodic_shift(class_id, item[1])]
        if not allowed:
            continue
        _, components, stats, _ = cv2.connectedComponentsWithStats((labels == class_id).astype(np.uint8))
        for component, (x, y, cw, ch, pixels) in enumerate(stats[1:], 1):
            if pixels < 64 or x < 1 or y < 1 or x+cw >= sw or y+ch >= sh:
                continue
            source_slice = np.s_[y-1:y+ch+1, x-1:x+cw+1]
            mask = (components[source_slice] == component).astype(np.uint8)
            halo = cv2.dilate(mask, kernel) > 0
            if (halo & (~source_known[source_slice]
                        | ((labels[source_slice] > 0) & (labels[source_slice] != class_id)))).any():
                continue
            corners = np.array([[[x-1, y-1], [x+cw+1, y-1],
                                 [x+cw+1, y+ch+1], [x-1, y+ch+1]]], np.float64)
            for offset, _, shift, C in allowed:
                projected = cv2.perspectiveTransform(corners, C)[0]
                if not np.isfinite(projected).all():
                    continue
                xx, yy = np.floor(projected.min(axis=0)).astype(int)-1
                right, bottom = np.ceil(projected.max(axis=0)).astype(int)+1
                if xx < 0 or yy < 0 or right > w or bottom > h:
                    continue
                size = (int(right-xx), int(bottom-yy))
                to_local = np.array([[1., 0., -xx], [0., 1., -yy], [0., 0., 1.]])
                from_source = np.array([[1., 0., x-1], [0., 1., y-1], [0., 0., 1.]])
                local = to_local @ C @ from_source
                mapped = cv2.warpPerspective(mask, local, size, flags=cv2.INTER_NEAREST) > 0
                dest_slice = np.s_[yy:bottom, xx:right]
                dest_halo = cv2.dilate(mapped.astype(np.uint8), kernel) > 0
                if not mapped.any() or (dest_halo & (~known[dest_slice] | occupied[dest_slice])).any():
                    continue
                if source_gray is None:
                    source_gray, target_gray = to_gray(source_image), to_gray(result.input_image)
                observed = cv2.warpPerspective(source_gray[source_slice], local, size,
                                               flags=cv2.INTER_NEAREST)
                raw = _ncc(observed[mapped].astype(np.float32),
                           target_gray[dest_slice][mapped].astype(np.float32))
                records['appearance_candidates'] += 1
                if not np.isfinite(raw) or raw < cfg.periodic_min_ncc:
                    continue
                if resize_region(MaskRegion(yy, xx, mapped)).sum() < 16:
                    continue
                proposals.append((raw, class_id, component, offset, shift, C, dest_slice,
                                  mapped, [int(x), int(y), int(x+cw), int(y+ch)]))
    if not proposals:
        return
    src = _features(*ValidRegion.resize_for_matching(source_gray, source_valid, ss))
    dst = _features(*ValidRegion.resize_for_matching(target_gray, target_valid, ts))
    local_cfg = replace(cfg, min_overlap=.2)
    warps = {}
    for raw, class_id, component, offset, shift, C, dest_slice, mapped, source_bbox in sorted(
            proposals, key=lambda proposal: -proposal[0]):
        if (mapped & occupied[dest_slice]).any():
            continue
        work = Db @ C @ np.linalg.inv(Da)
        key = tuple(offset)
        if key not in warps:
            warps[key] = ScoreWarp(src, work, ts)
        roi = cv2.dilate(resize_region(MaskRegion(dest_slice[0].start, dest_slice[1].start, mapped)),
                         np.ones((25, 25), np.uint8)) & dst[3]
        metrics = _score(src, (dst[0], dst[1], dst[2], roi), work, local_cfg, warped=warps[key])
        if not (metrics['structure_ncc'] >= cfg.periodic_min_ncc
                and metrics['edge_alignment'] >= cfg.periodic_min_edge):
            continue
        result.mapped_labels[dest_slice][mapped] = class_id
        occupied[dest_slice] |= mapped
        if result.label_periodic_valid is not None:
            result.label_periodic_valid[dest_slice] |= mapped
        ys, xs = np.where(mapped)
        bbox = [int(dest_slice[1].start+xs.min()), int(dest_slice[0].start+ys.min()),
                int(dest_slice[1].start+xs.max()+1), int(dest_slice[0].start+ys.max()+1)]
        count = int(mapped.sum())
        records['copies'].append(dict(class_id=class_id, template_index=template_index,
            source_component=component, source_bbox=source_bbox, target_bbox=bbox,
            matrix_template_to_target=C.tolist(), target_vector=shift.tolist(),
            offset=list(offset), raw_pixel_ncc=float(raw), metrics=metrics, added_label_pixels=count))
        records['added_label_pixels'] += count
    if result.label_periodic_valid is not None:
        info['added_label_pixels'] = int(result.label_periodic_valid.sum())
    info['boundary_foreground_pixels'] = (info.get('boundary_foreground_pixels', 0)
                                          + records['added_label_pixels'])
