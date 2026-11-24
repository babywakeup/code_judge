"""Recover isolated semantic instances using observed, verified target periods.

This is deliberately separate from unknown-seeded template transfer: a known
background island needs its own positive evidence, not a larger aligned copy.
Run after identity fusion so no existing foreground can be overwritten.
"""

from dataclasses import replace
from itertools import product

import cv2
import numpy as np

from ...utils import ValidRegion, to_gray
from ..registration.core import _features
from ..registration.scoring import ScoreWarp, _ncc, _score
from .reference_instances import complete_reference_instances
from .regions import MaskRegion, RegionResize


def complete_periodic_instances(result, target_valid, cfg, *, reference=None):
    """Verify observed contours first, then independent source annotations."""
    _complete_observed_instances(result, target_valid, cfg)
    if reference is not None:
        complete_reference_instances(result, target_valid, cfg, **reference)


def _complete_observed_instances(result, target_valid, cfg):
    """Copy only complete, frozen semantic contours into empty known regions.

    The seed's label comes from reference transfer, but appearance is compared
    between actual occurrences in the target. This avoids treating a different
    reference identity's local appearance as evidence for another small object.
    Neither newly filled contours nor ordered-identity masks can become seeds.
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
            offsets.append((offset, vectors, shift))
    if not offsets:
        return
    offsets.sort(key=lambda item: (sum(abs(n) for n in item[0]), item[0]))
    labels = result.mapped_labels
    # Freeze all evidence before any writes; accepted copies cannot bootstrap.
    frozen = labels.copy()
    known = result.label_valid & (target_valid > 0)
    foreground = frozen > 0
    classes = [int(c) for c in np.unique(frozen[known])
               if c > 0 and c != cfg.ordered_object_class and cfg.periodic_axes(int(c))]
    records = dict(copies=[], considered_components=0, appearance_candidates=0,
                   added_label_pixels=0)
    info['interior_instances'] = records
    gray = None
    proposals = []
    h, w = labels.shape
    scale = cfg.matching_scale(result.template_image.shape, result.input_image.shape)
    size = (round(w*scale), round(h*scale))
    D = np.diag([size[0]/w, size[1]/h, 1.])
    resize_region = RegionResize(labels.shape, size)
    kernel = np.ones((3, 3), np.uint8)
    for class_id in classes:
        allowed = [item for item in offsets if cfg.allows_periodic_shift(class_id, item[1])]
        if not allowed:
            continue
        _, components, stats, _ = cv2.connectedComponentsWithStats((frozen == class_id).astype(np.uint8))
        for component, (x, y, cw, ch, pixels) in enumerate(stats[1:], 1):
            if pixels < 64 or x < 1 or y < 1 or x+cw >= w or y+ch >= h:
                continue
            source_slice = np.s_[y-1:y+ch+1, x-1:x+cw+1]
            mask = (components[source_slice] == component).astype(np.uint8)
            halo = cv2.dilate(mask, kernel) > 0
            # Unknown boundaries and other-class occlusions do not establish a
            # complete source contour. Never propagate their clipped remnants.
            if (halo & (~known[source_slice]
                        | (foreground[source_slice] & (frozen[source_slice] != class_id)))).any():
                continue
            records['considered_components'] += 1
            for offset, _, shift in allowed:
                xx, yy = np.floor(np.array([x-1, y-1])+shift).astype(int)
                width, height = cw+3, ch+3
                if xx < 0 or yy < 0 or xx+width > w or yy+height > h:
                    continue
                local = np.array([[1., 0., x-1+shift[0]-xx],
                                  [0., 1., y-1+shift[1]-yy]])
                mapped = cv2.warpAffine(mask, local, (width, height), flags=cv2.INTER_NEAREST) > 0
                dest_slice = np.s_[yy:yy+height, xx:xx+width]
                dest_halo = cv2.dilate(mapped.astype(np.uint8), kernel) > 0
                if (dest_halo & (~known[dest_slice] | foreground[dest_slice])).any():
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
                                  mapped, [int(x), int(y), int(x+cw), int(y+ch)]))
    if not proposals:
        return
    # Equivalent x/y aliases compete for one destination. Highest own-contour
    # appearance wins; never union near-duplicate poses into an enlarged mask.
    proposals.sort(key=lambda item: -item[0])
    features = _features(*ValidRegion.resize_for_matching(gray, target_valid, size))
    local_cfg = replace(cfg, min_overlap=.2)
    occupied = foreground.copy()
    warps = {}
    for raw, class_id, component, offset, shift, dest_slice, mapped, source_bbox in proposals:
        if (mapped & occupied[dest_slice]).any():
            continue
        P = np.eye(3)
        P[:2, 2] = shift
        work = D @ P @ np.linalg.inv(D)
        key = tuple(shift)
        if key not in warps:
            warps[key] = ScoreWarp(features, work, size)
        roi = cv2.dilate(resize_region(MaskRegion(dest_slice[0].start, dest_slice[1].start, mapped)),
                         np.ones((25, 25), np.uint8)) & features[3]
        metrics = _score(features, (features[0], features[1], features[2], roi), work,
                         local_cfg, warped=warps[key])
        if not (metrics['structure_ncc'] >= cfg.periodic_min_ncc
            and metrics['edge_alignment'] >= cfg.periodic_min_edge):
            continue
        labels[dest_slice][mapped] = class_id
        occupied[dest_slice] |= mapped
        if result.label_periodic_valid is not None:
            result.label_periodic_valid[dest_slice] |= mapped
        yy, xx = np.where(mapped)
        bbox = [int(dest_slice[1].start+xx.min()), int(dest_slice[0].start+yy.min()),
                int(dest_slice[1].start+xx.max()+1), int(dest_slice[0].start+yy.max()+1)]
        count = int(mapped.sum())
        records['copies'].append(dict(class_id=class_id, source_component=component,
            source_target_bbox=source_bbox, target_bbox=bbox, offset=list(offset),
            target_vector=shift.tolist(), raw_pixel_ncc=float(raw), metrics=metrics,
            added_label_pixels=count))
        records['added_label_pixels'] += count
    if result.label_periodic_valid is not None:
        info['added_label_pixels'] = int(result.label_periodic_valid.sum())
    info['boundary_foreground_pixels'] = (info.get('boundary_foreground_pixels', 0)
                                          + records['added_label_pixels'])