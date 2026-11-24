"""Serial AB/BA comparison of a saved support implementation and current 640 code.

Loads the old module in memory; never swaps production files or reads target GT.
Only object-support completion changes. Initialization, warmup and PNG export are
excluded from measured inference. Both implementations use the SAME matcher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from periodic_template_matcher import PeriodicTemplateMatcher
from periodic_template_matcher import matcher as facade


def load_baseline(path):
    name = 'periodic_template_matcher._internal.objects._benchmark_baseline'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.complete_object_support


def compare(before, after):
    old, new = before.mapped_labels == 2, after.mapped_labels == 2
    support = after.registration['ordered_objects']['support_completion']
    unchanged = {}
    for name in ('matrix_template_to_target', 'object_valid', 'base_labels', 'base_valid',
                 'geometric_valid'):
        # Object validity is intentionally extended; the other arrays must not
        # change since registration, source selection and base transfer are fixed.
        unchanged[name] = bool(np.array_equal(getattr(before, name), getattr(after, name)))
    if not all(v for k, v in unchanged.items() if k != 'object_valid'):
        raise AssertionError(f'Unexpected upstream difference: {unchanged}')
    if (old & ~new).any():
        raise AssertionError('Previously accepted class-2 foreground was lost')
    if not np.array_equal(before.object_identities[old], after.object_identities[old]):
        raise AssertionError('Previously confirmed identity changed')
    return dict(unchanged=unchanged, class2_before=int(old.sum()), class2_after=int(new.sum()),
        recovered_foreground=int((new & ~old).sum()),
        lost_known_pixels=int((before.label_valid & ~after.label_valid).sum()),
        added_known_pixels=int((after.label_valid & ~before.label_valid).sum()),
        known_before=int(before.label_valid.sum()), known_after=int(after.label_valid.sum()),
        template_index=after.registration['ranking'][0],
        order=after.registration['ordered_objects']['visible_order'],
        support=support)


def overlay(result):
    image = result.input_image.copy()
    positive = result.mapped_labels == 2
    image[positive] = (.45*image[positive]+.55*np.array([0, 255, 0])).astype(np.uint8)
    return image


def benchmark(config, baseline_path, targets, output, rounds):
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(config.read_text(encoding='utf-8-sig'))
    for key in ('target', 'target_valid_mask'):
        if spec.get(key):
            spec[key] = str((config.parent/spec[key]).resolve())
    for template in spec['templates']:
        for key in ('image', 'labelme', 'labels', 'valid_mask', 'annotation_valid'):
            if template.get(key):
                template[key] = str((config.parent/template[key]).resolve())
    options = spec.setdefault('options', {})
    if 'template_workers' in options:
        options['template_workers'] = 1
    else:
        options.setdefault('step2', {})['template_workers'] = 1
    snapshot = output/'config.snapshot.json'
    snapshot.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    matcher = PeriodicTemplateMatcher(snapshot)
    if matcher.config.max_side != 640 or matcher.config.scale is not None:
        raise ValueError('This comparison requires max_side=640 and scale=null')
    baseline = load_baseline(baseline_path)
    current = facade.complete_object_support
    report = dict(config=str(config), baseline=str(baseline_path),
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        serial=True, template_workers=1, rounds=rounds,
        timing_scope='warm infer only; excludes initialization, warmup and export', targets=[])

    def run(target, implementation):
        support_seconds, ecc_calls = [], []
        real_ecc = cv2.findTransformECC
        def ecc(*args, **kwargs):
            ecc_calls.append(1)
            return real_ecc(*args, **kwargs)
        def complete(*args, **kwargs):
            started = time.perf_counter()
            implementation(*args, **kwargs)
            support_seconds.append(time.perf_counter()-started)
        with patch.object(facade, 'complete_object_support', complete), patch.object(cv2, 'findTransformECC', ecc):
            started = time.perf_counter()
            result = matcher.infer(target=target)
            elapsed = time.perf_counter()-started
        return result, dict(infer_seconds=elapsed, support_seconds=sum(support_seconds),
                            ecc_calls=len(ecc_calls))

    for index, target in enumerate(targets, 1):
        folder = output/f'target{index}'
        folder.mkdir()
        run(target, baseline)
        run(target, current)
        rows = dict(before=[], after=[])
        saved = {}
        for repeat in range(rounds):
            order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
            for mode in order:
                result, measurement = run(target, baseline if mode == 'before' else current)
                rows[mode].append(measurement)
                (folder/'timings.partial.json').write_text(
                    json.dumps(rows, indent=2), encoding='utf-8')
                print(json.dumps(dict(target=index, round=repeat+1, mode=mode, **measurement)), flush=True)
                if mode not in saved:
                    saved[mode] = result
        delta = compare(saved['before'], saved['after'])
        for mode, result in saved.items():
            matcher.save_result(result, folder/mode)
        height = spec['options'].get('step1', {}).get('valid_y_end') or saved['after'].input_image.shape[0]
        panels = []
        for mode in ('before', 'after'):
            panel = overlay(saved[mode])[:height]
            panel = cv2.resize(panel, (985, round(panel.shape[0]*985/panel.shape[1])))
            cv2.putText(panel, f'{mode} 640', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            panels.append(panel)
        cv2.imencode('.png', np.hstack(panels))[1].tofile(folder/'comparison.png')
        medians = {mode: {key: statistics.median(row[key] for row in values)
                         for key in ('infer_seconds', 'support_seconds', 'ecc_calls')}
                   for mode, values in rows.items()}
        report['targets'].append(dict(target=str(target), runs=rows, medians=medians, comparison=delta))
        (output/'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(target=index, medians=medians,
                              recovered_foreground=delta['recovered_foreground'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'periodic_template_matcher/config.sd1.labelme.opencv.json')
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--target', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    benchmark(args.config.resolve(), args.baseline.resolve(), [p.resolve() for p in args.target],
              args.output.resolve(), args.rounds)


if __name__ == '__main__':
    main()