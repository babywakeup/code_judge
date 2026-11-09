"""Compare registration concurrency with existing config/data; never writes results.

Run from the repository root: python -m tools.benchmark_template_parallelism
Process mode is an optional measurement, not a second production backend.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import statistics
from time import perf_counter

import cv2
import numpy as np

from periodic_template_matcher import MatcherConfig
from periodic_template_matcher._internal.registration.core import _prepare_image, _register_prepared
from periodic_template_matcher._internal.registration.correlation import _periods
from periodic_template_matcher.utils import ValidRegion, read_image


def _initialize_process():
    cv2.setNumThreads(2)


def _job(args):
    image, valid, cfg, scale, destination, lattice = args
    source = _prepare_image(image, valid, cfg, scale)
    return _register_prepared(source, destination, cfg, lattice)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('periodic_template_matcher/config.sd1.labelme.opencv.json'))
    parser.add_argument('--templates', type=int, default=4, help='Repeat available templates up to this count (default 4)')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--processes', action='store_true', help='Also compare a spawned process pool')
    args = parser.parse_args()
    if args.templates < 1 or args.repeats < 2:
        parser.error('templates must be positive; repeats must be >= 2 for cold/warm reporting')
    path = args.config.resolve()
    spec = json.loads(path.read_text(encoding='utf-8-sig'))
    resolve = lambda value: path.parent / value
    cfg = MatcherConfig.from_options(spec.get('options', {}))
    originals = [read_image(resolve(s['image'])) for s in spec['templates']]
    target = read_image(resolve(spec['target']))
    templates = [originals[i % len(originals)] for i in range(args.templates)]
    masks = []
    for i, image in enumerate(templates):
        s = spec['templates'][i % len(originals)]
        extra = read_image(resolve(s['valid_mask']), cv2.IMREAD_GRAYSCALE) if s.get('valid_mask') else None
        masks.append(ValidRegion.from_config(cfg, s.get('valid_height')).build_mask(image, extra))
    extra = read_image(resolve(spec['target_valid_mask']), cv2.IMREAD_GRAYSCALE) if spec.get('target_valid_mask') else None
    target_mask = ValidRegion.from_config(cfg, spec.get('target_valid_height')).build_mask(target, extra)
    cfg = replace(cfg, valid_height=None)
    cv2.setNumThreads(2)
    scale = cfg.matching_scale(*(im.shape for im in [*templates, target]))
    start = perf_counter()
    destination = _prepare_image(target, target_mask, cfg, scale)
    lattice = _periods(destination.features[0], destination.features[3]) if destination.features is not None else []
    common = perf_counter() - start
    print(json.dumps(dict(templates=args.templates, distinct_available=len(originals),
                          repeated=args.templates > len(originals), backend=cfg.backend,
                          working_scale=scale, common_target_preparation_s=common,
                          opencv_threads=2), indent=2), flush=True)
    jobs = [(im, valid, cfg, scale, destination, lattice) for im, valid in zip(templates, masks)]
    modes = [('serial', 1), ('thread', 2), ('thread', 4)]
    if args.processes:
        modes.append(('process', 4))
    baseline = None
    pools, timings, setup, shutdown = {}, {}, {}, {}
    # Interleave methods instead of exhausting all serial runs first: thermal
    # changes/background CPU load otherwise bias a fixed mode ordering.
    try:
        for mode, workers in modes:
            key = (mode, workers)
            start = perf_counter()
            pools[key] = (ThreadPoolExecutor(max_workers=workers) if mode == 'thread' else
                          ProcessPoolExecutor(max_workers=workers, initializer=_initialize_process)
                          if mode == 'process' else None)
            setup[key] = perf_counter() - start
            timings[key] = []
        for repeat in range(args.repeats):
            order = modes if repeat % 2 == 0 else list(reversed(modes))
            for mode, workers in order:
                key = (mode, workers)
                executor = pools[key]
                tick = perf_counter()
                results = list(executor.map(_job, jobs)) if executor else [_job(job) for job in jobs]
                elapsed = perf_counter() - tick
                timings[key].append(elapsed + (setup[key] if repeat == 0 else 0))
                if baseline is None:
                    baseline = results
                for expected, actual in zip(baseline, results):
                    if abs(expected['match_score'] - actual['match_score']) > 1e-7:
                        raise AssertionError('Serial/parallel score mismatch')
                    a, b = expected['matrix_template_to_target'], actual['matrix_template_to_target']
                    if a is None or b is None:
                        if a is not b:
                            raise AssertionError('Serial/parallel matrix availability mismatch')
                    else:
                        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
                print(json.dumps(dict(mode=mode, workers=workers, repeat=repeat,
                                      elapsed_s=elapsed)), flush=True)
    finally:
        for key, executor in pools.items():
            start = perf_counter()
            if executor:
                executor.shutdown()
            shutdown[key] = perf_counter() - start
    for (mode, workers), samples in timings.items():
        print(json.dumps(dict(mode=mode, workers=workers, runs_s=samples,
                              cold_s=samples[0], warm_median_s=statistics.median(samples[1:]),
                              median_with_common_s=statistics.median(samples) + common,
                              shutdown_s=shutdown[(mode, workers)],
                              total_active_s=sum(samples) + shutdown[(mode, workers)],
                              scores_and_matrices_match=True)), flush=True)


if __name__ == '__main__':
    main()
