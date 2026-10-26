"""Experimental periodic label transfer; target labels are evaluation-only.

Run from repository root: python -m examples.periodic_labelme_probe
This probe uses the saved global alignment and near-translation SD-1 geometry.
It is not a production periodic extrapolation backend.
"""
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

from periodic_template_matcher.utils import read_image, write_image, overlay_labels, warp_labels


def polygon_mask(item, shape):
    mask = np.zeros(shape, np.uint16)
    cv2.fillPoly(mask, [np.rint(item['points']).astype(np.int32)], 1)
    return mask


def iou(pred, truth, roi):
    a, b = (pred > 0) & roi, (truth > 0) & roi
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--period-source', choices=('labelme', 'image'), default='labelme')
    parser.add_argument('--no-local-refine', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / 'source/SD-1/10-21'
    suffix = args.period_source + ('_unrefined' if args.no_local_refine else '_refined')
    out = root / 'periodic_template_matcher/results/periodic_labelme_probe' / suffix
    out.mkdir(parents=True, exist_ok=True)
    source_path = data / 'Tlink-Image-1792571258237.jpeg'
    target_path = data / 'Tlink-Image-1792571352308_defect.jpeg'
    source, target = read_image(source_path), read_image(target_path)
    source_items = json.loads(source_path.with_suffix('.json').read_text(encoding='utf-8'))['shapes']
    source_items = [s for s in source_items if s['label'] == '1']
    shape = target.shape[:2]
    centers = np.array([(np.min(s['points'], axis=0) + np.max(s['points'], axis=0)) / 2
                        for s in source_items])
    # This dataset has two rows, each with three corresponding anchor regions.
    row_order = np.argsort(centers[:, 1])
    rows = [sorted(row_order[:3], key=lambda i: centers[i, 0]),
            sorted(row_order[3:], key=lambda i: centers[i, 0])]
    # Estimate periods from the retained columns only, not the held-out column.
    horizontal = np.median([centers[row[1]] - centers[row[0]] for row in rows], axis=0)
    vertical = np.median([centers[b] - centers[a] for a, b in zip(rows[0][:2], rows[1][:2])], axis=0)
    saved = root / 'periodic_template_matcher/results/sd1_labelme_defect/result.json'
    registration = json.loads(saved.read_text(encoding='utf-8'))
    H = np.array(registration['matches'][registration['ranking'][0]]['matrix_template_to_target'])
    if args.period_source == 'image':
        best = registration['matches'][registration['ranking'][0]]
        vector = next(c['vector_work_px'] for c in best['lattice_candidates']
                      if c['vector_work_px'][0] > 0)
        # Convert target working-image vector to source-original coordinates.
        working_size = best['working_target_size']
        target_vector = np.array(vector) * np.array([shape[1]/working_size[0], shape[0]/working_size[1]])
        horizontal = np.linalg.solve(H[:2, :2], target_vector)
    masks = [polygon_mask(item, source.shape[:2]) for item in source_items]
    sg = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    roi = np.ones(shape, bool)
    roi[int(shape[0] * .92):] = False
    full = np.maximum.reduce(masks)
    baseline, valid = warp_labels(full, H, shape)
    valid &= roi
    baseline[~valid] = 0
    # Generate periodic proposals from the retained first two columns.
    keep = [i for row in rows for i in row[:2]]
    retained = np.maximum.reduce([masks[i] for i in keep])
    held_baseline, _ = warp_labels(retained, H, shape)
    extended = baseline.copy()
    held_extended = held_baseline.copy()
    added = np.zeros(shape, np.uint8)
    proposals = []
    for row in rows:
        donor = row[1]
        pts = np.asarray(source_items[donor]['points'])
        lo = np.floor(pts.min(axis=0)).astype(int) - 45
        hi = np.ceil(pts.max(axis=0)).astype(int) + 46
        x0, y0 = np.maximum(lo, 0)
        x1, y1 = np.minimum(hi, [source.shape[1], source.shape[0]])
        patch = sg[y0:y1, x0:x1]
        # Include candidates beyond both image edges; only fully visible patches
        # are verifiable in this simple probe. Partial boundary units stay unknown.
        for n in (-2, -1, 1, 2):
            shift = n * horizontal
            predicted = H @ np.array([x0 + shift[0], y0 + shift[1], 1.])
            px, py = np.rint(predicted[:2] / predicted[2]).astype(int)
            ph, pw = patch.shape
            radius = 16
            sx, sy = max(0, px-radius), max(0, py-radius)
            ex, ey = min(shape[1], px+pw+radius), min(shape[0], py+ph+radius)
            if px < 0 or py < 0 or px+pw > shape[1] or py+ph > int(shape[0]*.92):
                proposals.append(dict(row=int(donor), period_offset=n, status='partial_or_outside'))
                continue
            scores = cv2.matchTemplate(tg[sy:ey, sx:ex], patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            actual = np.array([sx+location[0], sy+location[1]])
            delta = actual - predicted[:2] / predicted[2]
            proposals.append(dict(row=int(donor), period_offset=n, score=score,
                                  local_correction=delta.tolist(), status='accepted' if score >= .8 else 'rejected'))
            if score < .8:
                continue
            if args.no_local_refine:
                delta = np.zeros(2)
            P = np.eye(3)
            P[:2, 2] = shift
            C = np.eye(3)
            C[:2, 2] = delta
            mapped, _ = warp_labels(masks[donor], C @ H @ P, shape)
            foreground = (mapped > 0) & roi
            newly_covered = foreground & ~valid
            extended[newly_covered] = 1
            added[newly_covered] = 255
            held_extended[foreground] = 1
    # Only now read target polygons: never used for period estimation or search.
    truth_items = json.loads(target_path.with_suffix('.json').read_text(encoding='utf-8'))['shapes']
    truth = np.maximum.reduce([polygon_mask(s, shape) for s in truth_items if s['label'] == '1'])
    third_column_roi = roi.copy()
    third_column_roi[:, :1800] = False
    truth_missing = (truth > 0) & roi & ~valid
    result = dict(note='EXPERIMENT ONLY; translation-like alignment, two rows/three columns assumed; target labels evaluation-only',
                  horizontal_period_original_px=horizontal.tolist(), vertical_period_original_px=vertical.tolist(),
                  period_source=args.period_source, local_refine=not args.no_local_refine,
                  global_lattice_candidates=registration['matches'][0]['lattice_candidates'],
                  actual=dict(baseline_iou=iou(baseline, truth, roi), extended_iou=iou(extended, truth, roi),
                              uncovered_gt_foreground_pixels=int(truth_missing.sum()),
                              added_foreground_pixels=int((added > 0).sum())),
                  held_out_right_column=dict(baseline_iou=iou(held_baseline, truth, third_column_roi),
                                             periodic_iou=iou(held_extended, truth, third_column_roi),
                                             note='rightmost source polygons withheld; no target labels used for matching'),
                  proposals=proposals)
    write_image(out / 'baseline.jpg', overlay_labels(target, baseline, roi))
    write_image(out / 'held_out_before.jpg', overlay_labels(target, held_baseline, roi))
    write_image(out / 'held_out_after.jpg', overlay_labels(target, held_extended, roi))
    write_image(out / 'truth.jpg', overlay_labels(target, truth, roi))
    write_image(out / 'added_foreground.png', added)
    (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
