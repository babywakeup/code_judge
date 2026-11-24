"""Alternate serial full-frame/ROI ECC and report speed AND prediction drift.

No ground-truth accuracy claim is made. Initialization and export are excluded
from infer_seconds. All other options are shared, except template_workers=1.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from periodic_template_matcher import PeriodicTemplateMatcher


def array_snapshot(result):
    return {f.name: dict(shape=list(value.shape), dtype=str(value.dtype),
                        sha256=hashlib.sha256(value.tobytes()).hexdigest())
            for f in fields(result) if isinstance((value := getattr(result, f.name)), np.ndarray)}


def compare(reference, result):
    if reference.label_valid is None or result.label_valid is None:
        return dict(accepted_unchanged=result.accepted == reference.accepted,
                    labels_comparable=False,
                    reference_has_labels=reference.label_valid is not None,
                    roi_has_labels=result.label_valid is not None)
    both = reference.label_valid & result.label_valid
    classes = sorted(set(np.unique(reference.mapped_labels[both]).tolist())
                     | set(np.unique(result.mapped_labels[both]).tolist()))
    iou = {}
    for label in classes:
        a = (reference.mapped_labels == label) & both
        b = (result.mapped_labels == label) & both
        union = int((a | b).sum())
        iou[str(label)] = float((a & b).sum()/union) if union else None
    old_objects = reference.registration.get('ordered_objects', {})
    new_objects = result.registration.get('ordered_objects', {})
    transforms = []
    for old, new in zip(reference.registration['matches'], result.registration['matches']):
        a, b = old['matrix_template_to_target'], new['matrix_template_to_target']
        if a is not None and b is not None:
            transforms.append(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
    h, w = reference.input_image.shape[:2]
    points = np.array([[[0., 0.], [w-1., 0.], [w-1., h-1.], [0., h-1.]]], np.float32)
    def local_transforms(objects):
        # A reference may support multiple vertical occurrences of one object.
        # Keep the occurrence ordinal; a dict keyed only by reference loses data.
        return {(obj['identity'], index, ref['template_index'], ref['reference_index']):
                np.asarray(ref['matrix_template_to_target'])
            for obj in objects.get('objects', [])
            for index, ref in enumerate(obj.get('supporting_references', []))}
    old_local, new_local = local_transforms(old_objects), local_transforms(new_objects)
    local_errors = []
    for key in old_local.keys() & new_local.keys():
        delta = new_local[key] @ np.linalg.inv(old_local[key])
        local_errors.append(float(np.linalg.norm(cv2.perspectiveTransform(points, delta)-points, axis=2).max()))
    return dict(accepted_unchanged=result.accepted == reference.accepted,
                ranking_unchanged=result.registration['ranking'] == reference.registration['ranking'],
                order_unchanged=old_objects.get('visible_order') == new_objects.get('visible_order'),
                changed_known_labels=int(((reference.mapped_labels != result.mapped_labels) & both).sum()),
                shared_known_pixels=int(both.sum()),
                changed_valid_pixels=int((reference.label_valid != result.label_valid).sum()),
                changed_identity_pixels=int((reference.object_identities != result.object_identities).sum())
                    if reference.object_identities is not None and result.object_identities is not None else None,
                class_iou_on_shared_known=iou,
                global_matrix_max_abs_delta=max(transforms, default=None),
                local_reference_keys_unchanged=old_local.keys() == new_local.keys(),
                local_target_corner_max_displacement_original_px=max(local_errors, default=None))


def benchmark(config: Path, output: Path, targets: list[Path], repeats=3, *, option='ecc_roi'):
    if option not in ('ecc_roi', 'shared_periods'):
        raise ValueError('Only ecc_roi or shared_periods can be compared')
    output.mkdir(parents=True, exist_ok=False)
    config = config.resolve()
    spec = json.loads(config.read_text(encoding='utf-8-sig'))
    for key in ('target', 'target_valid_mask'):
        if spec.get(key):
            spec[key] = str((config.parent/spec[key]).resolve())
    for template in spec['templates']:
        for key in ('image', 'labelme', 'labels', 'valid_mask', 'annotation_valid'):
            if template.get(key):
                template[key] = str((config.parent/template[key]).resolve())
    options = spec.setdefault('options', {})
    options.pop('template_workers', None)
    options.setdefault('step2', {})['template_workers'] = 1
    options.pop(option, None)
    matchers = {}
    modes = ('full', 'roi') if option == 'ecc_roi' else ('independent', 'shared')
    group = 'step3' if option == 'ecc_roi' else 'periodic'
    for name, enabled in zip(modes, (False, True)):
        options.setdefault(group, {})[option] = enabled
        path = output/f'config.{name}.json'
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        matchers[name] = PeriodicTemplateMatcher(path)
    targets = [p.resolve() for p in targets] or [Path(spec['target'])]
    rows = []
    for target_index, target in enumerate(targets):
        for repeat in range(repeats):
            results, records = {}, {}
            for name in (modes if repeat % 2 == 0 else modes[::-1]):
                start = time.perf_counter()
                result = matchers[name].infer(target=target)
                elapsed = time.perf_counter()-start
                results[name] = result
                row = dict(target=str(target), mode=name, run=repeat+1, infer_seconds=elapsed,
                           accepted=result.accepted, arrays=array_snapshot(result))
                rows.append(row)
                records[name] = row
                print(json.dumps({k: v for k, v in row.items() if k != 'arrays'}), flush=True)
                if repeat == 0:
                    matchers[name].save_result(result, output/f'target{target_index+1}.{name}')
            drift = compare(results[modes[0]], results[modes[1]])
            records[modes[1]]['difference_from_full' if option == 'ecc_roi' else 'difference_from_independent'] = drift
            print(json.dumps(drift), flush=True)
        summary = {name: statistics.median(r['infer_seconds'] for r in rows
                   if r['mode'] == name and r['target'] == str(target)) for name in matchers}
        print(json.dumps(dict(target=str(target), median_infer_seconds=summary)), flush=True)
    report = dict(option=option, timing_scope='warm_infer_only_excluding_initialization_hashing_and_export', rows=rows)
    (output/'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'periodic_template_matcher/config.sd1.labelme.opencv.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=Path, action='append', default=[])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    benchmark(args.config, args.output, args.target, args.repeats)


if __name__ == '__main__':
    main()