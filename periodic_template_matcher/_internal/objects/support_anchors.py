"""Bounded, identity-preserving anchors for verified periodic support.

A primary occurrence at a photograph boundary is not a canonical lattice origin.
Interior defects may hide appearance, but cannot supply an alignment anchor.
"""
import cv2
import numpy as np

from ...utils import ValidRegion, to_gray, warp_valid_mask
from ..registration.scoring import _ncc
from ..segmentation.regions import MaskRegion, component_regions
from .layout import is_partial_reference


def periodic_affine(H, source, target, pivot, corners, D):
    """Least-change affine obeying L*source=target about an observed pivot.

    A short visible fragment can give good local ECC but a slightly wrong
    periodic slope. Correct only the verified lattice direction, preserve the
    observed pivot, and retain the existing six-work-pixel displacement bound.
    This is a proposal, not acceptance or an additional ECC solve.
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    if not np.allclose(H[2], [0., 0., 1.]) or np.dot(source, source) < 1:
        return None
    linear = H[:2, :2]
    fitted = linear+np.outer(target-linear@source, source)/np.dot(source, source)
    displacement = (corners-pivot) @ (fitted-linear).T @ D[:2, :2].T
    if np.linalg.norm(displacement, axis=1).max() > 6.:
        return None
    result = H.copy()
    result[:2, :2] = fitted
    result[:2, 2] += (linear-fitted) @ pivot
    if not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-10:
        return None
    return result


def canonical_support_objects(prediction, images, references, source_valids, vectors,
                              target_gray, work_size, cfg, *, target_valid):
    """One best in-canvas alias per identity/direction, never a new ECC search.

    Rank only against the ORIGINAL confirmed pixels. Bridged fills cannot
    promote themselves into identity evidence or start an unbounded walk.
    The full-resolution kernel still verifies every proposed fill afterwards.
    """
    if not cfg.shared_periods:
        return []
    lattice = prediction.diagnostics.get('periodic_lattice', [])
    verified = [a for a in lattice if a.get('verified')]
    lattice_template = prediction.diagnostics.get('periodic_lattice_template_index',
                                                  0 if len(images) == 1 else None)
    size = work_size
    D = np.diag([size[0]/target_gray.shape[1], size[1]/target_gray.shape[0], 1.])
    labels = cv2.resize(prediction.identities, size, interpolation=cv2.INTER_NEAREST)
    valid = cv2.resize(prediction.valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    target, target_roi = ValidRegion.resize_for_matching(target_gray, target_valid, size)
    valid &= target_roi > 0
    center = (np.array(target_gray.shape[::-1])-1)/2
    prepared = {}
    result = []
    for obj in prediction.diagnostics.get('objects', []):
        identity = list(cfg.object_cycle).index(obj['identity'])+1
        if not ((labels == identity) & valid).any():
            continue
        anchors = [obj, *obj.get('supporting_references', [])]
        for vector in vectors:
            axis = next((a for a in verified if np.allclose(vector, a['target_vector'])), None)
            if axis is None:
                continue
            best = None
            seen = set()
            for anchor in anchors:
                if anchor is obj and obj.get('reference_partial', False):
                    continue
                ti, ri = anchor['template_index'], anchor['reference_index']
                ref = references[ti][ri]
                if ref['identity'] != obj['identity']:
                    continue
                key = ti, ri
                if key not in prepared:
                    raw = ((ref['mask'] > 0) & (source_valids[ti] > 0)).astype(np.uint8)
                    yy, xx = np.where(raw)
                    if not len(xx) or is_partial_reference(ref, int(xx.min()), int(xx.max()), raw.shape[1]):
                        prepared[key] = None
                    else:
                        source_center = np.array([(xx.min()+xx.max())/2, (yy.min()+yy.max())/2, 1.])
                        corners = np.array([[x, y] for x in (xx.min(), xx.max())
                                            for y in (yy.min(), yy.max())], float)
                        prepared[key] = raw, source_center, corners, to_gray(images[ti])
                if prepared[key] is None:
                    continue
                raw, source_center, corners, gray = prepared[key]
                H = np.array(anchor['matrix_template_to_target'], dtype=float)
                # Only rephase a genuinely observed occurrence. An arbitrary
                # off-canvas transform is not an unlimited extrapolation seed.
                original = warp_valid_mask(raw, D @ H, target.shape)
                original_own = original & valid & (labels == identity)
                if int(original_own.sum()) < 64:
                    continue
                proposals = [(H, False)]
                if ti == lattice_template:
                    yy, xx = np.where(original_own)
                    pivot = np.linalg.inv(D @ H) @ [xx.mean(), yy.mean(), 1.]
                    fitted = periodic_affine(H, axis['source_vector'], vector,
                                             pivot[:2]/pivot[2], corners, D)
                    if fitted is not None:
                        proposals.append((fitted, True))
                for pose, constrained in proposals:
                    H = pose.copy()
                    mapped_center = H @ source_center
                    offset = int(round(np.dot(center-mapped_center[:2]/mapped_center[2], vector)
                                       / np.dot(vector, vector)))
                    H[:2, 2] += offset*vector
                    signature = ti, ri, tuple(np.round(H.ravel(), 6))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    mask = warp_valid_mask(raw, D @ H, target.shape)
                    known = mask & valid
                    own = known & (labels == identity)
                    count = int(own.sum())
                    if count < 64 or count < cfg.object_min_score*int(known.sum()):
                        continue
                    observed = cv2.warpPerspective(gray, D @ H, size)
                    ncc = _ncc(observed[own].astype(np.float32), target[own].astype(np.float32))
                    quality = ncc, count/max(int(known.sum()), 1), count
                    if best is None or quality > best[0]:
                        # Do not inherit source-specific metadata or mutable lists
                        # from a primary reference in a DIFFERENT template.
                        record = dict(identity=obj['identity'], template_index=ti, reference_index=ri,
                                      reference_partial=False,
                                      matrix_template_to_target=H.tolist(),
                                      support_anchor=dict(offset=offset, target_vector=vector.tolist(),
                                          lattice_constrained=constrained, known_ncc=ncc, known_pixels=count))
                        best = quality, record
            if best is not None:
                result.append(best[1])
    return result


def interior_bridges(mapped, prediction, identity, observed, target_gray, occluded,
                     max_gap, anchor_height, verify):
    """Bridge a connected source contour only between independently good ends.

    The missing interior is NOT declared appearance-matched. Both bracketing
    strips must pass the unchanged pixel-NCC, structural-NCC and edge gates.
    Only foreground is transferred; open ends, other identities, background and
    disconnected source components cannot be inferred by this rule.
    """
    approved = np.zeros(mapped.shape, bool)
    records = []
    for component in component_regions(mapped > 0):
        slices = component.slices
        own = ((prediction.identities[slices] == identity) & prediction.valid[slices])
        missing = component.mask & ~prediction.valid[slices]
        # Inspect actual pixel holes, not whole-image row occupancy. A surviving
        # pixel elsewhere in the row cannot hide a local missing contour.
        for gap in component_regions(missing):
            start, end = gap.y, gap.y+gap.mask.shape[0]
            left, right = gap.x, gap.x+gap.mask.shape[1]
            if start == 0 or end == len(component.mask) or end-start > max_gap:
                continue
            anchors = []
            remaining = max_gap-(end-start)
            for above in (True, False):
                # The nearest confirmed mask can itself cross the appearance
                # defect. Search OUTWARD for a healthy anchor, but keep the
                # separation within one configured period budget. This is a
                # bounded strip verification, never another image/ECC search.
                for offset in range(0, int(remaining)+1, anchor_height):
                    lo = max(0, start-anchor_height-offset) if above else end+offset
                    hi = start-offset if above else min(len(component.mask), lo+anchor_height)
                    if lo >= hi:
                        break
                    local = (slice(lo, hi), slice(left, right))
                    ys = slice(component.y+lo, component.y+hi)
                    xs = slice(component.x+left, component.x+right)
                    pixels = component.mask[local]
                    ids = prediction.identities[ys, xs]
                    # Do not jump over a conflicting identity to find a nicer
                    # patch further away in the same source component.
                    if (pixels & prediction.valid[ys, xs] & (ids > 0) & (ids != identity)).any():
                        break
                    visible = pixels & own[local] & ~occluded[ys, xs]
                    a = observed[ys, xs][visible].astype(np.float32)
                    b = target_gray[ys, xs][visible].astype(np.float32)
                    if (len(a) < max(64, .5*int(pixels.sum())) or np.std(a) < 2 or np.std(b) < 2):
                        continue
                    evidence = MaskRegion(component.y+lo, component.x+left, visible)
                    metrics = verify(evidence, _ncc(a, b))
                    if metrics is None:
                        continue
                    anchors.append(dict(target_bbox=[component.x+left, component.y+lo,
                        component.x+right, component.y+hi], pixels=len(a), metrics=metrics))
                    remaining -= offset
                    break
                if len(anchors) != (1 if above else 2):
                    break
            if len(anchors) != 2:
                continue
            # The SAME source component supplies both independently checked
            # ends. Never let a large matching island certify a smaller one.
            ys = slice(component.y+start, component.y+end)
            xs = slice(component.x+left, component.x+right)
            approved[ys, xs] |= gap.mask
            records.append(dict(target_bbox=[component.x+left, component.y+start,
                component.x+right, component.y+end], pixels=gap.pixels, anchors=anchors))
    return approved, records