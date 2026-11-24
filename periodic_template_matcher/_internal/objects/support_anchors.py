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
            bridge_only = None
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
                    if count < 64:
                        continue
                    sparse = count < cfg.object_min_score*int(known.sum())
                    if sparse and (known & (labels > 0) & (labels != identity)).any():
                        continue
                    observed = cv2.warpPerspective(gray, D @ H, size)
                    ncc = _ncc(observed[own].astype(np.float32), target[own].astype(np.float32))
                    quality = ncc, count/max(int(known.sum()), 1), count
                    selected = bridge_only if sparse else best
                    if selected is None or quality > selected[0]:
                        # Do not inherit source-specific metadata or mutable lists
                        # from a primary reference in a DIFFERENT template.
                        record = dict(identity=obj['identity'], template_index=ti, reference_index=ri,
                                      reference_partial=False,
                                      matrix_template_to_target=H.tolist(),
                                      support_anchor=dict(offset=offset, target_vector=vector.tolist(),
                                          lattice_constrained=constrained, known_ncc=ncc, known_pixels=count))
                        if sparse:
                            # Blank pixels transferred from a partial occurrence
                            # are weaker than two locally verified foreground ends.
                            record['support_anchor']['bridge_only'] = True
                            bridge_only = quality, record
                        else:
                            best = quality, record
            if best is None:
                best = bridge_only
            if best is not None:
                result.append(best[1])
    return result


def interior_bridges(mapped, prediction, identity, observed, target_gray, occluded,
                     max_gap, anchor_height, verify, *, phase_retry=False,
                     include_background=False, clipped_candidates=None, columns=slice(None)):
    """Bridge a connected source contour only between independently good ends.

    The missing interior is NOT declared appearance-matched. Both bracketing
    strips must pass the unchanged pixel-NCC, structural-NCC and edge gates.
    A clipped end with a verified inner anchor may be recorded separately for
    corroboration by another occurrence. It is never approved here.
    """
    approved = np.zeros(mapped.shape, bool)
    records = []
    x_offset = int(columns.start or 0)
    for local_component in component_regions(mapped[:, columns] > 0):
        component = MaskRegion(local_component.y, local_component.x + x_offset, local_component.mask)
        slices = component.slices
        own = ((prediction.identities[slices] == identity) & prediction.valid[slices])
        missing = component.mask & (prediction.identities[slices] == 0 if include_background
                                    else ~prediction.valid[slices])
        # Inspect actual pixel holes, not whole-image row occupancy. A surviving
        # pixel elsewhere in the row cannot hide a local missing contour.
        for gap in component_regions(missing):
            start, end = gap.y, gap.y+gap.mask.shape[0]
            left, right = gap.x, gap.x+gap.mask.shape[1]
            if end-start > max_gap or (phase_retry and gap.pixels < 64):
                continue
            clipped = 0
            if clipped_candidates is not None and end-start >= anchor_height:
                # A few surviving foreground pixels can separate a defect
                # from the actual crop edge without forming a viable anchor.
                margin = max(1, anchor_height // 8)
                if component.y == 0 and start <= margin:
                    clipped = -1
                elif slices[0].stop == mapped.shape[0] and len(component.mask)-end <= margin:
                    clipped = 1
            if not clipped and (start == 0 or end == len(component.mask)):
                continue
            anchors = []
            remaining = max_gap-(end-start)
            sides = (False, True) if clipped < 0 else (True, False)
            for side, above in enumerate(sides, 1):
                # The nearest confirmed mask can itself cross the appearance
                # defect. Search OUTWARD for a healthy anchor, but keep the
                # separation within one configured period budget. This is a
                # bounded strip verification, never another image/ECC search.
                stride = max(8, anchor_height // 2) if phase_retry else anchor_height
                for offset in range(0, int(remaining)+1, stride):
                    lo = max(0, start-anchor_height-offset) if above else end+offset
                    hi = start-offset if above else min(len(component.mask), lo+anchor_height)
                    if lo >= hi:
                        break
                    ys = slice(component.y+lo, component.y+hi)
                    # A curved contour can leave the gap's x extent before
                    # reaching its healthy end. Follow the connected source
                    # component within this strip.
                    xs = slices[1]
                    pixels = component.mask[lo:hi]
                    ids = prediction.identities[ys, xs]
                    # Do not jump over a conflicting identity to find a nicer
                    # patch further away in the same source component.
                    if (pixels & prediction.valid[ys, xs] & (ids > 0) & (ids != identity)).any():
                        break
                    visible = pixels & own[lo:hi] & ~occluded[ys, xs]
                    a = observed[ys, xs][visible].astype(np.float32)
                    b = target_gray[ys, xs][visible].astype(np.float32)
                    if (len(a) < max(64, .5*int(pixels.sum())) or np.std(a) < 2 or np.std(b) < 2):
                        continue
                    evidence = MaskRegion(component.y+lo, component.x, visible)
                    metrics = verify(evidence, _ncc(a, b))
                    if metrics is None:
                        continue
                    anchors.append(dict(target_bbox=[component.x, component.y+lo,
                        xs.stop, component.y+hi], pixels=len(a), metrics=metrics))
                    remaining -= offset
                    break
                if len(anchors) != side:
                    break
            if len(anchors) != 2 and not (clipped and len(anchors) == 1):
                continue
            # The SAME source component supplies both independently checked
            # ends. Never let a large matching island certify a smaller one.
            ys = slice(component.y+start, component.y+end)
            xs = slice(component.x+left, component.x+right)
            if len(anchors) == 1:
                clipped_candidates.append((MaskRegion(ys.start, xs.start, gap.mask), clipped, anchors[0]))
                continue
            approved[ys, xs] |= gap.mask
            if clipped < 0:
                anchors.reverse()
            records.append(dict(target_bbox=[component.x+left, component.y+start,
                component.x+right, component.y+end], pixels=gap.pixels, anchors=anchors))
    return approved, records


def corroborate_clipped_bridges(candidates, bridged, confirmed, identity, vector, tolerance):
    """Corroborate one clipped end with a two-anchor bridge one period away."""
    if not candidates or bridged is None:
        return
    seed = bridged.astype(np.uint8)
    support = (bridged | (confirmed.valid & (confirmed.identities == identity))).astype(np.uint8)
    kernel = np.ones((2*tolerance+1, 2*tolerance+1), np.uint8)
    for region, step, anchor in candidates:
        translation = np.eye(2, 3)
        translation[:, 2] = step*vector - [region.x, region.y]
        size = region.mask.shape[::-1]
        adjacent_bridge = cv2.warpAffine(seed, translation, size, flags=cv2.INTER_NEAREST) > 0
        if np.count_nonzero(region.mask & adjacent_bridge) < .5*region.pixels:
            continue
        adjacent = cv2.warpAffine(support, translation, size, flags=cv2.INTER_NEAREST)
        # One working pixel accounts for rasterization and lattice residual.
        agreed = region.mask & (cv2.dilate(adjacent, kernel) > 0)
        if np.count_nonzero(agreed) >= .85*region.pixels:
            yield MaskRegion(region.y, region.x, agreed), step, anchor


def verified_repeat_regions(fill, mapped, confirmed, identity, gray, occluded, vector,
                            tile_size, min_ncc, verify):
    """Use a confirmed occurrence of the same identity as a second source.

    Candidate pixels must agree with the canonical contour and the translated
    label. Each copied region must also match the target's actual appearance.
    """
    approved = np.zeros_like(fill)
    candidates = fill & (mapped > 0)
    seed = ((confirmed.identities == identity) & confirmed.valid & ~occluded).astype(np.uint8)
    records = []
    for step in (-1, 1):
        translation = np.eye(2, 3)
        translation[:, 2] = step * vector
        support = cv2.warpAffine(seed, translation, fill.shape[::-1], flags=cv2.INTER_NEAREST) > 0
        support &= candidates & ~approved
        if np.count_nonzero(support) < 64:
            continue
        for region in component_regions(support):
            if region.pixels < 64:
                continue
            local = translation.copy()
            local[:, 2] -= [region.x, region.y]
            observed = cv2.warpAffine(gray, local, region.mask.shape[::-1])
            target = gray[region.slices]
            visible = ~occluded[region.slices]

            def check(y, x, mask):
                slices = np.s_[y:y+mask.shape[0], x:x+mask.shape[1]]
                evidence = mask & visible[slices]
                if evidence.sum() < max(64, .5 * mask.sum()):
                    return False
                ncc = _ncc(observed[slices][evidence].astype(np.float32),
                           target[slices][evidence].astype(np.float32))
                if ncc < min_ncc:
                    return False
                metrics = verify(MaskRegion(region.y+y, region.x+x, evidence), ncc, step)
                if metrics is None:
                    return False
                out = approved[region.slices]
                out[slices] |= mask
                records.append(dict(step=step, target_bbox=[region.x+x, region.y+y,
                    region.x+x+mask.shape[1], region.y+y+mask.shape[0]],
                    pixels=int(mask.sum()), metrics=metrics))
                return True

            if check(0, 0, region.mask) or max(region.mask.shape) <= tile_size:
                continue
            for y in range(0, region.mask.shape[0], tile_size):
                for x in range(0, region.mask.shape[1], tile_size):
                    check(y, x, region.mask[y:y+tile_size, x:x+tile_size])
    return approved, records
