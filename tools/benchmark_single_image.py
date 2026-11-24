"""Serial, lossless before/after measurements for a configured matcher image.

This does not change thresholds or image resolution. Every run stores all result
array hashes and diagnostics, so a speedup cannot hide changed predictions.
Wall time excludes PNG export. Profiling covers initialization, inference and hashing.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from periodic_template_matcher import PeriodicTemplateMatcher


def benchmark(config: Path, output: Path, repeats=1, profile=False, compare: Path | None = None,
              target: Path | None = None, reuse_matcher=False):
    output.mkdir(parents=True, exist_ok=False)
    config = config.resolve()
    spec = json.loads(config.read_text(encoding='utf-8-sig'))
    for key in ('target', 'target_valid_mask'):
        if spec.get(key):
            spec[key] = str((config.parent / spec[key]).resolve())
    for template in spec['templates']:
        for key in ('image', 'labelme', 'labels', 'valid_mask', 'annotation_valid'):
            if template.get(key):
                template[key] = str((config.parent / template[key]).resolve())
    if target:
        spec['target'] = str(target.resolve())
    # Strictly serial registration as well as serial images; execution policy
    # changes only, never thresholds or image resolution.
    options = spec.setdefault('options', {})
    if 'template_workers' in options:
        options['template_workers'] = 1
    else:
        options.setdefault('step2', {})['template_workers'] = 1
    snapshot_config = output / 'config.snapshot.json'
    snapshot_config.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    snapshots = []

    def capture(result):
        arrays = {}
        for field in fields(result):
            value = getattr(result, field.name)
            if isinstance(value, np.ndarray):
                arrays[field.name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                         sha256=hashlib.sha256(value.tobytes()).hexdigest())
        # Compare the serialized public contract: config tuples become JSON
        # lists on disk, which is not a numerical/prediction difference.
        snapshots.append(json.loads(json.dumps(dict(arrays=arrays, registration=result.registration))))

    times, image_times = [], []
    profiler = cProfile.Profile() if profile else None
    matcher = None
    for index in range(repeats):
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        try:
            if matcher is None or not reuse_matcher:
                matcher = PeriodicTemplateMatcher(snapshot_config)
            result = matcher.infer()
            capture(result)
        finally:
            if profiler:
                profiler.disable()
        timings = {}
        manifest = matcher.save_result(result, output / f'run{index + 1}', timings=timings)
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        image_times.append(timings['image_save_seconds'])
        times.append(time.perf_counter() - start - image_times[-1])
        print(json.dumps(dict(run=index + 1, seconds=times[-1], accepted=metadata['accepted'],
                              image_save_seconds=image_times[-1],
                              template=metadata['best_template_index']), ensure_ascii=False), flush=True)
    report = dict(config=str(config.resolve()), timing_scope='excluding_image_save',
                  reuse_matcher=reuse_matcher, first_run_includes_initialization=True, seconds=times,
                  image_save_seconds=image_times, snapshots=snapshots)
    if profiler:
        profiler.dump_stats(str(output / 'profile.pstats'))
        stats = pstats.Stats(profiler)
        stages = {}
        for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
            if name in ('_coarse_candidates', '_refine', '_score', '_periods',
                        'refine_object_transform', 'match_ordered_objects', 'complete_periodic') or 'findTransformECC' in name:
                stages[f'{Path(filename).name}:{line}:{name}'] = dict(
                    calls=calls, own_seconds=own, cumulative_seconds=cumulative)
        report['profile_stages'] = stages
        stats.strip_dirs().sort_stats('cumulative').print_stats(35)
    (output / 'benchmark.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if compare:
        before = json.loads((compare / 'benchmark.json').read_text(encoding='utf-8'))
        expected = before['snapshots'][0]
        for index, snapshot in enumerate(snapshots):
            if snapshot != expected:
                changed = [key for key in snapshot['arrays']
                           if snapshot['arrays'][key] != expected['arrays'].get(key)]
                raise AssertionError(f'Run {index + 1}: changed arrays={changed}; '
                                     f'diagnostics_equal={snapshot["registration"] == expected["registration"]}')
        print('All result arrays and registration diagnostics are EXACTLY unchanged.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'periodic_template_matcher/config.sd1.labelme.opencv.json')
    parser.add_argument('--output', type=Path, required=True, help='A NEW directory; existing data is never overwritten')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--reuse-matcher', action='store_true',
                        help='Reuse prepared templates after run 1, like a gallery queue; run 1 still includes initialization')
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--target', type=Path, help='Optional source image override; the original config is never changed')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    benchmark(args.config, args.output, args.repeats, args.profile, args.compare, args.target, args.reuse_matcher)


if __name__ == '__main__':
    main()
