"""Identity-aware appearance matching for cyclic groups of different objects.

No label map is periodically tiled here. Each candidate comes from an original
object reference and actual visible target pixels; order only resolves candidates.
"""
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

from ..utils import ValidRegion, to_gray, warp_valid_mask
from .registration import _features


@dataclass
class ObjectPrediction:
    labels: np.ndarray
    identities: np.ndarray
    valid: np.ndarray
    diagnostics: dict


def _quality(candidate):
    """Prefer whole vertical structures over a coincidental short end fragment.

    Horizontal clipping at the input edge remains allowed and lightly penalized;
    it must not make an unrelated full-width identity win solely by area.
    """
    return (candidate['score']-.2*(1-candidate['vertical_fraction'])
            -.025*(1-candidate['horizontal_fraction'])-.05*candidate['reference_partial'])


def _correlation_peaks(patch, mask, target, valid, cfg):
    """Masked zero-mean NCC on each actual intersection, including image edges."""
    ph, pw = patch.shape
    patch = np.ascontiguousarray(patch*mask, np.float32)
    mask = np.ascontiguousarray(mask, np.float32)
    valid = np.ascontiguousarray(valid, np.float32)
    target = np.ascontiguousarray(target*valid, np.float32)

    def corr(a, b):
        return cv2.matchTemplate(cv2.copyMakeBorder(a, ph-1, ph-1, pw-1, pw-1,
                                  cv2.BORDER_CONSTANT, value=0), b, cv2.TM_CCORR)

    count = np.maximum(corr(valid, mask), 0)
    st, ss = corr(target, mask), corr(valid, patch)
    et, es = corr(target*target, mask), corr(valid, patch*patch)
    cross = corr(target, patch)
    denominator = np.maximum(count, 1)
    numerator = cross - st*ss/denominator
    energy = np.maximum(et-st*st/denominator, 0)*np.maximum(es-ss*ss/denominator, 0)
    score = numerator/np.sqrt(np.maximum(energy, 1e-8))
    eligible = ((count >= max(64, cfg.object_min_visible*float(mask.sum()))) & (energy > 1e-6))
    score = np.where(eligible & np.isfinite(score), np.clip(score, -1, 1), -1)
    peaks = []
    for _ in range(cfg.object_candidates):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        value = float(score[y, x])
        if value < cfg.object_min_score:
            break
        peaks.append((value, x-pw+1, y-ph+1))
        rx, ry = max(6, pw//3), max(6, ph//8)
        score[max(0, y-ry):y+ry+1, max(0, x-rx):x+rx+1] = -1
    return peaks


def match_ordered_objects(images, references, source_valids, target, target_valid, matrices, cfg):
    """Match three independent identities in a coherent cyclic arrangement."""
    shape = target.shape[:2]
    labels = np.zeros(shape, np.uint16)
    identities = np.zeros(shape, np.uint16)
    known = np.zeros(shape, bool)
    cycle = list(cfg.object_cycle)
    info = dict(status='insufficient_evidence', visible_order='', identity_names={str(i+1): n for i, n in enumerate(cycle)},
                objects=[], candidate_counts={}, alternatives=[], valid_coverage_roi=0.)
    scale = cfg.matching_scale(target.shape, *(im.shape for im in images))
    ts = (round(shape[1]*scale), round(shape[0]*scale))
    Db = np.diag([ts[0]/shape[1], ts[1]/shape[0], 1.])
    tg, tm = ValidRegion.resize_for_matching(to_gray(target), target_valid, ts)
    tf = _features(tg, tm)[0]
    pools = {name: [] for name in cycle}
    for ti, (image, refs, source_valid, initial) in enumerate(zip(images, references, source_valids, matrices)):
        if initial is None:
            continue
        ss = (round(image.shape[1]*scale), round(image.shape[0]*scale))
        Da = np.diag([ss[0]/image.shape[1], ss[1]/image.shape[0], 1.])
        sg, sm = ValidRegion.resize_for_matching(to_gray(image), source_valid, ss)
        sf = _features(sg, sm)[0]
        linear = Db @ np.asarray(initial) @ np.linalg.inv(Da)
        linear[:2, 2] = 0
        corners = cv2.perspectiveTransform(np.array([[[0., 0.], [ss[0]-1., 0.],
                           [ss[0]-1., ss[1]-1.], [0., ss[1]-1.]]], np.float32), linear)[0]
        low, high = np.floor(corners.min(axis=0)), np.ceil(corners.max(axis=0))
        linear[:2, 2] -= low
        canvas_size = tuple((high-low+1).astype(int))
        aligned = cv2.warpPerspective(sf, linear, canvas_size)
        aligned_valid = cv2.warpPerspective(sm, linear, canvas_size, flags=cv2.INTER_NEAREST)
        for ri, ref in enumerate(refs):
            raw = ((ref['mask'] > 0) & (source_valid > 0)).astype(np.uint8)
            if not raw.any():
                continue
            mask = cv2.resize(raw, ss, interpolation=cv2.INTER_NEAREST)
            mask = cv2.warpPerspective(mask, linear, canvas_size, flags=cv2.INTER_NEAREST)
            context = cv2.dilate(mask, np.ones((9, 9), np.uint8)) & aligned_valid
            yy, xx = np.where(context > 0)
            if len(xx) < 64:
                continue
            x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
            patch, weight = aligned[y0:y1, x0:x1], context[y0:y1, x0:x1]
            raw_ys, raw_xs = np.where(raw > 0)
            reference_partial = (raw_xs.min() < 2 or raw_xs.max() >= image.shape[1]-2)
            center = np.array([[[float((raw_xs.min()+raw_xs.max())/2),
                                float((raw_ys.min()+raw_ys.max())/2)]]], np.float32)
            for score, dx, dy in _correlation_peaks(patch, weight, tf, tm, cfg):
                shift = np.eye(3)
                shift[:2, 2] = [dx-x0, dy-y0]
                H = np.linalg.inv(Db) @ shift @ linear @ Da
                visible = warp_valid_mask(raw, H, target.shape) & (target_valid > 0)
                fraction = min(1., float(visible.sum()/max(float(raw.sum())*abs(np.linalg.det(H[:2, :2])), 1)))
                if fraction < cfg.object_min_visible or visible.sum() < 64:
                    continue
                vertical_fraction = min(1., float(np.any(visible, axis=1).sum()/max(
                    (raw_ys.max()-raw_ys.min()+1)*np.linalg.norm(H[:2, 1]), 1)))
                cx = float(cv2.perspectiveTransform(center, H)[0, 0, 0])
                pools[ref['identity']].append(dict(identity=ref['identity'], template_index=ti,
                    reference_index=ri, shape_index=ref.get('shape_index', ri), score=score,
                    reference_partial=bool(reference_partial),
                    visible_fraction=fraction, partial=fraction < .95, center_x=cx,
                    vertical_fraction=vertical_fraction,
                    horizontal_fraction=min(1., fraction/max(vertical_fraction, 1e-6)),
                    matrix_template_to_target=H.tolist(), _mask=visible,
                    _support=warp_valid_mask(source_valid, H, target.shape),
                    _background_known=ref.get('background_known', True)))
    for name in cycle:
        candidates = sorted(pools[name], key=_quality, reverse=True)
        distinct = []
        for candidate in candidates:
            # Retain one best reference per horizontal location; vertical variants
            # are kept separately for same-identity support after order resolution.
            if all(abs(candidate['center_x']-prev['center_x']) > max(8., 12/scale) for prev in distinct):
                distinct.append(candidate)
            if len(distinct) >= cfg.object_candidates:
                break
        pools[name] = dict(all=candidates, distinct=distinct)
        info['candidate_counts'][name] = len(candidates)
    groups = []
    for phase in range(3):
        order = cycle[phase:]+cycle[:phase]
        for group in product(*(pools[name]['distinct'] for name in order)):
            if any(a['center_x'] >= b['center_x'] for a, b in zip(group, group[1:])):
                continue
            if any((a['_mask'] & b['_mask']).sum() > .15*min(a['_mask'].sum(), b['_mask'].sum())
                   for a, b in zip(group, group[1:])):
                continue
            groups.append((sum(_quality(c) for c in group)/3, order, group))
    groups.sort(key=lambda group: group[0], reverse=True)
    if not groups:
        return ObjectPrediction(labels, identities, known, info)
    score, order, chosen = groups[0]
    alternative = next((g for g in groups[1:] if g[1] != order), None)
    margin = score-alternative[0] if alternative is not None else None
    info.update(visible_order=''.join(order), score=score, order_margin=margin,
                alternatives=[dict(order=''.join(o), score=s) for s, o, _ in groups[:6]],
                proposed_objects=[{k: v for k, v in c.items() if not k.startswith('_')} for c in chosen])
    if margin is not None and margin < cfg.object_min_gap:
        info['status'] = 'ambiguous_identity'
        return ObjectPrediction(labels, identities, known, info)
    info['status'] = 'accepted'
    known = target_valid > 0
    for chosen_object in chosen:
        name = chosen_object['identity']
        selected = [c for c in pools[name]['all'] if abs(c['center_x']-chosen_object['center_x']) < 8/scale
                    and c['score'] >= chosen_object['score']-.025]
        combined = np.zeros(shape, bool)
        support = np.zeros(shape, bool)
        for candidate in selected:
            combined |= candidate['_mask']
            support |= candidate['_support']
        # Missing vertical source support cannot be invented at target boundaries.
        ys, xs = np.where(combined)
        band = np.zeros(shape, bool)
        band[:, xs.min():xs.max()+1] = True
        known &= ~(band & ~support)
        if not chosen_object['_background_known']:
            known &= combined
        if all(c['reference_partial'] for c in selected):
            # A fragment says where observed foreground is, not how far the
            # unseen object continues beyond the source photograph.
            known &= combined
            info['status'] = 'partial_reference'
        labels[combined] = cfg.ordered_object_class
        identities[combined] = cycle.index(name)+1
        record = {k: v for k, v in chosen_object.items() if not k.startswith('_')}
        record['supporting_transforms'] = [c['matrix_template_to_target'] for c in selected]
        info['objects'].append(record)
    known |= labels > 0
    known &= target_valid > 0
    info['valid_coverage_roi'] = float(known.sum()/max(int((target_valid > 0).sum()), 1))
    return ObjectPrediction(labels, identities, known, info)
