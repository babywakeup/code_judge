"""Recover isolated semantic instances using observed, verified target periods.

This is deliberately separate from unknown-seeded template transfer: a known
background island needs its own positive evidence, not a larger aligned copy.
The periodic layer is built against the semantic evidence before object fusion.
"""

from dataclasses import replace
from itertools import product

import cv2
import numpy as np

from ...utils import ValidRegion, to_gray, warp_labels
from ..registration.core import _features
from ..registration.scoring import ScoreWarp, _ncc, _score
from .reference_instances import complete_reference_instances
from .layers import record_periodic_prediction
from .regions import MaskRegion, RegionResize


def complete_periodic_instances(result, target_valid, cfg, *, reference=None, alternate_references=()):
    """Complete instances from independently verified template annotations."""
    if reference is not None:
        complete_reference_instances(result, target_valid, cfg, **reference)
        matrix = result.matrix_template_to_target
        lattice = result.registration.get('periodic_completion', {}).get('lattice')
        if matrix is not None and lattice:
            valid = reference['source_valid'] > 0
            if reference['annotation_valid'] is not None:
                valid &= reference['annotation_valid'] > 0
            donors = warp_labels(reference['labels'], matrix, target_valid.shape, valid)
            _complete_observed_instances(result, target_valid, cfg, donors=donors)
    for alternate in alternate_references:
        complete_reference_instances(result, target_valid, cfg, **alternate)


def _complete_observed_instances(result, target_valid, cfg, *, donors=None):
    """Copy complete semantic contours into independently verified locations.

    The pipeline supplies projected template labels as contour donors. The
    target's peer occurrence provides appearance evidence for the translation.
    """
    if not (cfg.periodic_completion and cfg.shared_periods and cfg.label_mode == 'semantic'):
        return
    info = result.registration.get('periodic_completion', {})
    lattice = info.get('lattice', [])
    if not lattice or result.mapped_labels is None or result.label_valid is None:
        return
    offsets = []
    for offset in product(range(-cfg.periodic_max_steps, cfg.periodic_max_steps+1), repeat=len(lattice)):
        active = [axis for n, axis in zip(offset, lattice) if n]
        if not active or not all(axis.get('verified', False) for axis in active):
            continue
        vectors = [np.asarray(axis['source_vector'], float) for axis in active]
        shift = sum((n*np.asarray(axis['target_vector'], float)
                     for n, axis in zip(offset, lattice)), np.zeros(2))
        if np.isfinite(shift).all() and np.linalg.norm(shift) >= 1:
            target_verified = all(axis.get('target_verification', {}).get('verified', False)
                                  for axis in active)
            offsets.append((offset, vectors, shift, target_verified))
    if not offsets:
        return
    offsets.sort(key=lambda item: (sum(abs(n) for n in item[0]), item[0]))
    labels = result.mapped_labels
    # Freeze all evidence before any writes; accepted copies cannot bootstrap.
    frozen = labels.copy()
    target_roi = target_valid > 0
    known = result.label_valid & target_roi
    if result.base_valid is not None and result.object_valid is not None:
        # Fusion withholds background where the ordered object is unresolved.
        # The base layer can still establish a complete semantic contour.
        known |= result.base_valid & ~result.object_valid & target_roi
    outside_footprint = target_roi & ~known & ~result.geometric_valid
    foreground = frozen > 0
    source_labels, source_known = (frozen, known) if donors is None else donors
    source_known = source_known & target_roi
    classes = [int(c) for c in np.unique(source_labels[source_known])
               if c > 0 and c != cfg.ordered_object_class and cfg.periodic_axes(int(c))]
    records = dict(copies=[], considered_components=0, appearance_candidates=0,
                   added_label_pixels=0)
    info['interior_instances'] = records
    gray = None
    proposals = []
    h, w = labels.shape
    scale = cfg.matching_scale(result.template_image.shape, result.input_image.shape)
    edge_margin = max(1, int(np.ceil(1/scale)))
    size = (round(w*scale), round(h*scale))
    D = np.diag([size[0]/w, size[1]/h, 1.])
    resize_region = RegionResize(labels.shape, size)
    kernel = np.ones((3, 3), np.uint8)
    seed_halo = np.ones((2*edge_margin+1, 2*edge_margin+1), np.uint8)
    for class_id in classes:
        allowed = [item for item in offsets if cfg.allows_periodic_shift(class_id, item[1])]
        if not allowed:
            continue
        _, components, stats, _ = cv2.connectedComponentsWithStats((source_labels == class_id).astype(np.uint8))
        for component, (x, y, cw, ch, pixels) in enumerate(stats[1:], 1):
            if (pixels < cfg.min_contour_pixels or x < edge_margin or y < edge_margin
                    or x+cw > w-edge_margin or y+ch > h-edge_margin):
                continue
            source_slice = np.s_[y-1:y+ch+1, x-1:x+cw+1]
            mask = (components[source_slice] == component).astype(np.uint8)
            clearance = np.s_[y-edge_margin:y+ch+edge_margin, x-edge_margin:x+cw+edge_margin]
            seed = components[clearance] == component
            halo = cv2.dilate(seed.astype(np.uint8), seed_halo) > 0
            # Unknown boundaries and other-class occlusions do not establish a
            # complete source contour. Never propagate their clipped remnants.
            local_labels = source_labels[clearance]
            if (halo & (~source_known[clearance]
                        | ((local_labels > 0) & (local_labels != class_id)))).any():
                continue
            direct_seed = donors is not None
            if donors is None and result.direct_labels is not None and result.direct_valid is not None:
                direct = result.direct_labels[clearance] == class_id
                if (seed & direct).any():
                    if (halo & ~result.direct_valid[clearance]).any():
                        continue
                    direct_seed = not (seed & ~direct).any()
            records['considered_components'] += 1
            for offset, _, shift, target_verified in allowed:
                xx, yy = np.floor(np.array([x-1, y-1])+shift).astype(int)
                right, bottom = xx+cw+3, yy+ch+3
                can_extend = direct_seed and target_verified
                clipped = xx < 0 or yy < 0 or right > w or bottom > h
                if clipped and not can_extend:
                    continue
                xx, yy, right, bottom = max(0, xx), max(0, yy), min(w, right), min(h, bottom)
                if xx >= right or yy >= bottom:
                    continue
                width, height = int(right-xx), int(bottom-yy)
                local = np.array([[1., 0., x-1+shift[0]-xx],
                                  [0., 1., y-1+shift[1]-yy]])
                mapped = cv2.warpAffine(mask, local, (width, height), flags=cv2.INTER_NEAREST) > 0
                dest_slice = np.s_[yy:bottom, xx:right]
                dest_halo = cv2.dilate(mapped.astype(np.uint8), kernel) > 0
                fill = mapped
                if clipped or (dest_halo & (~known[dest_slice] | foreground[dest_slice])).any():
                    if not can_extend:
                        continue
                    fill = _footprint_extension(mapped, dest_halo, frozen[dest_slice], known[dest_slice],
                                                outside_footprint[dest_slice], class_id)
                    if fill is None:
                        continue
                if gray is None:
                    gray = to_gray(result.input_image)
                # Discrete native pixels, not one-sided interpolation smoothing.
                # Subpixel geometry is checked separately by structural scoring.
                observed = cv2.warpAffine(gray[source_slice], local, (width, height),
                                          flags=cv2.INTER_NEAREST)
                raw = _ncc(observed[mapped].astype(np.float32),
                           gray[dest_slice][mapped].astype(np.float32))
                records['appearance_candidates'] += 1
                if not np.isfinite(raw) or raw < cfg.periodic_min_ncc:
                    continue
                if resize_region(MaskRegion(yy, xx, mapped)).sum() < 16:
                    continue
                proposals.append((raw, class_id, component, offset, shift, dest_slice,
                                  mapped, fill, [int(x), int(y), int(x+cw), int(y+ch)]))
    if not proposals:
        return
    # Equivalent x/y aliases compete for one destination. Highest own-contour
    # appearance wins; never union near-duplicate poses into an enlarged mask.
    proposals.sort(key=lambda item: -item[0])
    features = _features(*ValidRegion.resize_for_matching(gray, target_valid, size), cfg=cfg)
    local_cfg = replace(cfg, min_overlap=cfg.local_min_overlap)
    occupied = foreground.copy()
    warps = {}
    newly_covered = 0
    for raw, class_id, component, offset, shift, dest_slice, mapped, fill, source_bbox in proposals:
        if (fill & occupied[dest_slice]).any():
            continue
        P = np.eye(3)
        P[:2, 2] = shift
        work = D @ P @ np.linalg.inv(D)
        key = tuple(shift)
        if key not in warps:
            warps[key] = ScoreWarp(features, work, size)
        roi = cv2.dilate(resize_region(MaskRegion(dest_slice[0].start, dest_slice[1].start, mapped)),
                         np.ones((cfg.local_context_size,)*2, np.uint8)) & features[3]
        metrics = _score(features, (features[0], features[1], features[2], roi), work,
                         local_cfg, warped=warps[key])
        if not (metrics['structure_ncc'] >= cfg.periodic_min_ncc
            and metrics['edge_alignment'] >= cfg.periodic_min_edge):
            continue
        prediction = np.full(mapped.shape, class_id, dtype=labels.dtype)
        record_periodic_prediction(result, prediction, mapped & target_roi[dest_slice], region=dest_slice)
        labels[dest_slice][fill] = class_id
        valid = result.label_valid[dest_slice]
        newly_covered += int(np.count_nonzero(fill & ~valid))
        valid |= fill
        occupied[dest_slice] |= fill
        if result.label_periodic_valid is not None:
            result.label_periodic_valid[dest_slice] |= fill
        yy, xx = np.where(fill)
        bbox = [int(dest_slice[1].start+xx.min()), int(dest_slice[0].start+yy.min()),
                int(dest_slice[1].start+xx.max()+1), int(dest_slice[0].start+yy.max()+1)]
        count = len(xx)
        records['copies'].append(dict(class_id=class_id, source_component=component,
            source_target_bbox=source_bbox, target_bbox=bbox, offset=list(offset),
            target_vector=shift.tolist(), raw_pixel_ncc=float(raw), metrics=metrics,
            added_label_pixels=count))
        records['added_label_pixels'] += count
    if result.label_periodic_valid is not None:
        info['added_label_pixels'] = int(result.label_periodic_valid.sum())
    info['boundary_foreground_pixels'] = (info.get('boundary_foreground_pixels', 0)
                                          + records['added_label_pixels'] - newly_covered)
    if newly_covered:
        records['newly_covered_pixels'] = newly_covered
        total = int(np.count_nonzero(target_valid))
        missing = int(np.count_nonzero((target_valid > 0) & ~result.label_valid))
        info.update(status='incomplete' if missing else 'complete',
                    remaining_unknown_pixels=missing, label_coverage_roi=(total-missing)/total)


def _footprint_extension(mapped, halo, labels, known, outside, class_id):
    """Continue a verified direct contour only into the unobserved footprint edge."""
    fill = mapped & outside
    if not fill.any() or (mapped & ~(known | outside)).any():
        return None
    if (halo & (labels > 0) & (labels != class_id)).any():
        return None
    overlap = mapped & known
    # A clipped contour may already have a visible portion. Require that it
    # agrees with the projection and keep its original pixels.
    if np.count_nonzero(overlap & (labels == class_id)) < .75*np.count_nonzero(overlap):
        return None
    return fill
