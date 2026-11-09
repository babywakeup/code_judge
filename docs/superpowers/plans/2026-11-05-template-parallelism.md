# Template parallelism implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. No changes to matching algorithms or acceptance policy.

**Goal:** Reduce latency of one target matched against up to four templates.

**Architecture:** Use a bounded ThreadPoolExecutor around independent source preparation and registration. Prepare the target/lattice once and share read-only; consume results in input order before stable ranking. Avoid nested executors. Keep the existing OpenCV two-thread policy outside the worker tasks.

**Tech Stack:** Python standard library concurrent.futures, existing NumPy/OpenCV.

## Design and alternatives

- Threads share original arrays and target features. Local four-task probe: 1.735 s median versus serial 3.809 s, with identical scores.
- Windows processes require spawn/import and serialized arrays. Probe with four processes: 1.831 s median including a cold run and two warm runs; no shared-memory optimization tested. Difference is workload-specific, not a universal claim about processes.
- Choose threads for this four-template, single-target workload. Default `template_workers=0` automatically caps workers at min(template count, CPU count, 4); `1` selects serial. Positive explicit values remain bounded by template count and CPU count. OpenCV remains at the existing two native threads, configured before launching tasks, never inside them. Existing OpenCV process-wide setting is not a per-matcher isolation guarantee.
- Single-template calls do not create an executor. Exceptions propagate; all tasks are joined by the context manager. Preserve template indexes, stable ties, duplicate candidates, diagnostics, and input arrays.
- Parallelism applies to registration, not object matching or label completion. Whole-pipeline speedup is lower. Four real distinct templates are not yet supplied: repeat the two existing templates only in the benchmark, clearly labeled.

## Files and checks

- [x] Add `tests/test_template_parallelism.py`: config validation, worker bounds, exact serial/thread results for FFT/OpenCV, deterministic ties, read-only input, overlapping task execution, single-template fallback, propagated failure.
- [x] Run failing tests, then add the config field in `config.py` and bounded parallel selection in `registration/core.py`.
- [x] Add `tools/benchmark_template_parallelism.py` for 1/2/4 workers and process comparison with repetitions; report common setup, cold/warm execution and consistent scores/matrices. Keep process option benchmark-only.
- [x] Update README and the SD1 OpenCV config to show the switch. Run new tests and full unittest suite; investigate and disclose rotation failures.
- [x] Benchmark actual current config end to end as well as registration; review the focused changes.

## Constraints

Do not commit or overwrite unrelated user modifications. No extra dependencies. No cache invalidation policy, process lifecycle API, candidate pruning, threshold or precision changes in this task.

## Verification ledger

- New tests: six passed; initially failed because `template_workers` did not exist.
- The first synthetic equivalence fixture used exact self-correlation. Investigation reproduced secondary symmetric-peak rounding differences across serial calls, independent of threading. Use a translated target and independent second template to test exact serial/parallel equivalence without this existing numerical tie. No scoring or peak-selection algorithm was altered.
- Fresh reviewer: no Important/Critical findings. Minor: distinguish warm reused pools from per-call API latency; documentation now states this.
- Full suite: 148 tests; the real SD1 rotation test failed its `label_valid[roi].all()` assertion for both cw/ccw. An isolated rerun of that test with `_template_worker_count` forced to 1 reproduced both failures at the same assertion (98.273 s). These failures also occur with parallel registration disabled; do not claim all tests pass.
- A benchmark run overlapped the full test suite and is excluded from reported performance. Rerun without other tests or pipeline workloads.
- Complete pipeline with actual two-template config: serial vs auto produced all 14 arrays pixel-exact and identical diagnostics (except concurrency config), for three calls each. Observed auto median 16.844 s; serial median 29.167 s was contaminated by the overlapping full suite, so do not report that ratio as speedup. Final uncontended serial call was 18.367 s.
- Standalone benchmark exhibited substantial runtime drift even without our other workloads. Interleaved/reversed method ordering and reported all samples instead of selecting best timings. Four-task cold kernel: serial 3.512 s, 2 threads 2.159 s, 4 threads 1.636 s, 4 processes 1.888 s. Common target setup 0.390 s. Later runs roughly doubled for all parent modes; process timings 1.817–3.086 s. These data support bounded threads as a low-overhead default, not a claim that threads always beat persistent processes. Additional distinct templates and production load need remeasurement.
- Benchmark environment: Windows, 20 logical CPUs, Python 3.12.13, installed NumPy 2.5.4/OpenCV 4.14.0. No dependencies were changed; these installed versions differ from the pinned OpenCV requirement, so report local measurements with that context.
