"""Serial AB/BA resolution comparison; output agreement is NOT GT accuracy."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

try:
    from .benchmark_ecc_roi import ROOT, PeriodicTemplateMatcher, array_snapshot, compare
except ImportError:
    from benchmark_ecc_roi import ROOT, PeriodicTemplateMatcher, array_snapshot, compare


def describe(result):
    registration = result.registration
    best = registration['matches'][registration['ranking'][0]]
    objects = registration.get('ordered_objects', {})
    periodic = registration.get('periodic_completion', {})
    return dict(accepted=result.accepted, status=registration.get('status'),
                template=registration.get('best_template_index'), score=best['match_score'],
                order=objects.get('visible_order'), object_status=objects.get('status'),
                known_pixels=int(result.label_valid.sum()) if result.label_valid is not None else 0,
                periodic_status=periodic.get('status'),
                remaining_unknown_pixels=periodic.get('remaining_unknown_pixels'),
                lattice_verified=[axis['verified'] for axis in periodic.get('lattice', [])])


def benchmark(config: Path, output: Path, targets: list[Path], *, reference=800, candidate=640, repeats=3):
    if repeats < 1 or min(reference, candidate) < 96 or reference == candidate:
        raise ValueError('Use positive repeats and two different resolutions >=96')
    config = config.resolve()
    original = json.loads(config.read_text(encoding='utf-8-sig'))
    for key in ('target', 'target_valid_mask'):
        if original.get(key):
            original[key] = str((config.parent/original[key]).resolve())
    for template in original['templates']:
        for key in ('image', 'labelme', 'labels', 'valid_mask', 'annotation_valid'):
            if template.get(key):
                template[key] = str((config.parent/template[key]).resolve())
    options = original.setdefault('options', {})
    if options.get('scale') is not None or options.get('step1', {}).get('scale') is not None:
        raise ValueError('An explicit scale overrides max_side; clear it before comparing')
    options.pop('template_workers', None)
    options.pop('max_side', None)
    options.setdefault('step2', {})['template_workers'] = 1
    output.mkdir(parents=True, exist_ok=False)
    modes = (str(reference), str(candidate))
    matchers = {}
    for side in (reference, candidate):
        spec = deepcopy(original)
        spec['options'].setdefault('step1', {})['max_side'] = side
        path = output/f'config.{side}.json'
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
        matchers[str(side)] = PeriodicTemplateMatcher(path)
    targets = [p.resolve() for p in targets] or [Path(original['target'])]
    report = dict(reference=reference, candidate=candidate, ground_truth_used=False,
                  timing_scope='reused_matcher_infer_excluding_initialization_hashing_export',
                  template_workers=1, rows=[], summaries=[])
    for target_index, target in enumerate(targets):
        target_rows = []
        for repeat in range(repeats):
            results, records = {}, {}
            for mode in (modes if repeat % 2 == 0 else modes[::-1]):
                start = time.perf_counter()
                result = matchers[mode].infer(target=target)
                elapsed = time.perf_counter()-start
                results[mode] = result
                row = dict(target=str(target), max_side=int(mode), run=repeat+1,
                           infer_seconds=elapsed, **describe(result))
                print(json.dumps(row, ensure_ascii=False), flush=True)
                row['arrays'] = array_snapshot(result)
                records[mode] = row
                target_rows.append(row)
                report['rows'].append(row)
                if repeat == 0:
                    matchers[mode].save_result(result, output/f'target{target_index+1}.{mode}')
            drift = compare(results[modes[0]], results[modes[1]])
            # A different resolution can reorder periodic appearances. The
            # occurrence ordinal is not a geometric correspondence across runs.
            drift.pop('local_target_corner_max_displacement_original_px', None)
            drift['local_transform_comparison'] = 'not_evaluated_occurrence_correspondence_not_guaranteed'
            a, b = results[modes[0]].label_valid, results[modes[1]].label_valid
            if a is not None and b is not None:
                drift.update(lost_known_pixels=int((a & ~b).sum()), gained_known_pixels=int((b & ~a).sum()))
            records[modes[1]]['difference_from_reference'] = drift
            print(json.dumps(drift, ensure_ascii=False), flush=True)
            (output/'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        summary = dict(target=str(target), median_infer_seconds={mode: statistics.median(
            row['infer_seconds'] for row in target_rows if row['max_side'] == int(mode)) for mode in modes})
        report['summaries'].append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        (output/'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'periodic_template_matcher/config.sd1.labelme.opencv.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', type=Path, action='append', default=[])
    parser.add_argument('--reference', type=int, default=800)
    parser.add_argument('--candidate', type=int, default=640)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    benchmark(args.config, args.output, args.target, reference=args.reference,
              candidate=args.candidate, repeats=args.repeats)


if __name__ == '__main__':
    main()