"""Serial warm AB/BA comparison of verified interior-instance fill on/off.

Only the new final stage is disabled for 'before'; all upstream code, prepared
templates, configuration and source assets are identical. No target GT is read.
"""

import argparse
import hashlib
import inspect
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
from periodic_template_matcher._internal import pipeline


def compare(before, after):
    unchanged = {name: bool(np.array_equal(getattr(before, name), getattr(after, name)))
                 for name in ('matrix_template_to_target', 'object_identities', 'object_valid',
                              'base_labels', 'base_valid', 'label_valid', 'geometric_valid',
                              'completed_template', 'completed_valid', 'periodic_valid')}
    if not all(unchanged.values()):
        raise AssertionError(f'Unexpected upstream difference: {unchanged}')
    old = before.mapped_labels > 0
    if not np.array_equal(before.mapped_labels[old], after.mapped_labels[old]):
        raise AssertionError('Previously accepted foreground was changed')
    if not np.array_equal(before.mapped_labels == 2, after.mapped_labels == 2):
        raise AssertionError('Class-2 foreground changed')
    copies = after.registration['periodic_completion'].get('interior_instances', {})
    return dict(unchanged=unchanged, class1_before=int((before.mapped_labels == 1).sum()),
        class1_after=int((after.mapped_labels == 1).sum()),
        recovered_foreground=int(((after.mapped_labels > 0) & ~old).sum()),
        known_pixels=int(after.label_valid.sum()),
        template_index=after.registration['ranking'][0],
        order=after.registration['ordered_objects']['visible_order'], interior_instances=copies,
        reference_instances=after.registration['periodic_completion'].get('reference_instances', {}))


def overlay(result):
    image = result.input_image.copy()
    for class_id, color in ((1, [0, 255, 255]), (2, [0, 200, 0])):
        mask = result.mapped_labels == class_id
        image[mask] = (.45*image[mask]+.55*np.asarray(color)).astype(np.uint8)
    return image


def benchmark(config, targets, output, rounds=3, *, baseline=None):
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(config.read_text(encoding='utf-8-sig'))
    if spec.get('target_valid_mask'):
        raise ValueError('A single-image target mask cannot be reused for these targets')
    spec['target'] = str(targets[0])
    for template in spec['templates']:
        for key in ('image', 'labelme', 'labels', 'valid_mask', 'annotation_valid'):
            if template.get(key):
                template[key] = str((config.parent/template[key]).resolve())
    spec['options'].setdefault('step2', {})['template_workers'] = 1
    snapshot = output/'config.snapshot.json'
    snapshot.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    matcher = PeriodicTemplateMatcher(snapshot)
    if matcher.config.max_side != 640 or matcher.config.scale is not None:
        raise ValueError('This comparison requires max_side=640 and scale=null')
    complete = pipeline.complete_periodic_instances
    source = inspect.getsource(inspect.getmodule(complete))
    (output/'implementation.txt').write_text(source, encoding='utf-8')
    from periodic_template_matcher._internal.segmentation import reference_instances
    reference_source = inspect.getsource(reference_instances)
    (output/'reference_implementation.txt').write_text(reference_source, encoding='utf-8')
    previous = None
    if baseline is not None:
        namespace = dict(__name__='periodic_template_matcher._internal.segmentation._benchmark_observed',
                         __package__='periodic_template_matcher._internal.segmentation')
        baseline_source = baseline.read_text(encoding='utf-8')
        exec(compile(baseline_source, str(baseline), 'exec'), namespace)
        previous = namespace['complete_periodic_instances']
    report = dict(serial=True, template_workers=1, rounds=rounds,
        implementation_sha256=hashlib.sha256(source.encode('utf-8')).hexdigest(),
        reference_implementation_sha256=hashlib.sha256(reference_source.encode('utf-8')).hexdigest(),
        baseline=('Saved observed-contour implementation without reference fallback' if previous else
                  'Same pipeline with only complete_periodic_instances disabled'),
        timing_scope='Warm infer only; excludes preparation, warmup and export', targets=[])
    if previous is not None:
        report.update(baseline_path=str(baseline),
                      baseline_sha256=hashlib.sha256(baseline_source.encode('utf-8')).hexdigest())

    def run(target, enabled):
        times, calls = [], []
        real_ecc = cv2.findTransformECC
        def ecc(*args, **kwargs):
            calls.append(1)
            return real_ecc(*args, **kwargs)
        def fill(*args, **kwargs):
            start = time.perf_counter()
            if enabled:
                complete(*args, **kwargs)
            elif previous is not None:
                previous(*args)
            times.append(time.perf_counter()-start)
        with patch.object(pipeline, 'complete_periodic_instances', fill), patch.object(cv2, 'findTransformECC', ecc):
            start = time.perf_counter()
            result = matcher.infer(target=target)
            elapsed = time.perf_counter()-start
        return result, dict(infer_seconds=elapsed, instance_seconds=sum(times), ecc_calls=len(calls))

    for index, target in enumerate(targets, 1):
        folder = output/f'target{index}'
        folder.mkdir()
        run(target, False)
        run(target, True)
        rows, saved = dict(before=[], after=[]), {}
        for repeat in range(rounds):
            for mode in (('before', 'after') if repeat % 2 == 0 else ('after', 'before')):
                result, measurement = run(target, mode == 'after')
                rows[mode].append(measurement)
                saved.setdefault(mode, result)
                (folder/'timings.partial.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
                print(json.dumps(dict(target=index, round=repeat+1, mode=mode, **measurement)), flush=True)
        delta = compare(saved['before'], saved['after'])
        for mode, result in saved.items():
            matcher.save_result(result, folder/mode)
        height = spec['options'].get('step1', {}).get('valid_y_end') or result.input_image.shape[0]
        panels = [cv2.resize(overlay(saved[mode])[:height], (985, round(height*985/result.input_image.shape[1])))
                  for mode in ('before', 'after')]
        for mode, panel in zip(('before', 'after'), panels):
            cv2.putText(panel, mode+' 640', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
        cv2.imencode('.png', np.hstack(panels))[1].tofile(folder/'comparison.png')
        crops = []
        all_copies = (delta['interior_instances'].get('copies', [])
                  + delta['reference_instances'].get('copies', []))
        for copy in all_copies:
            x0, y0, x1, y1 = copy['target_bbox']
            x0, y0 = max(0, x0-24), max(0, y0-24)
            x1, y1 = min(result.input_image.shape[1], x1+24), min(height, y1+24)
            panels = [cv2.resize(image[y0:y1, x0:x1], (240, 240)) for image in (
                result.input_image, overlay(saved['before']), overlay(saved['after']))]
            crops.append(np.hstack(panels))
        if crops:
            cv2.imencode('.png', np.vstack(crops))[1].tofile(folder/'instance_crops.png')
        medians = {mode: {key: statistics.median(row[key] for row in values)
                         for key in ('infer_seconds', 'instance_seconds', 'ecc_calls')}
                   for mode, values in rows.items()}
        report['targets'].append(dict(target=str(target), runs=rows, medians=medians, comparison=delta))
        (output/'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(target=index, medians=medians,
                              recovered_foreground=delta['recovered_foreground'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'periodic_template_matcher/config.sd1.labelme.opencv.json')
    parser.add_argument('--target', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--baseline', type=Path, help='Saved observed-only module source; loaded in memory')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    benchmark(args.config.resolve(), [p.resolve() for p in args.target], args.output.resolve(), args.rounds,
              baseline=None if args.baseline is None else args.baseline.resolve())


if __name__ == '__main__':
    main()