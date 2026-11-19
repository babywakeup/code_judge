# Reference Anomaly Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate general anomaly evidence from an existing corrected normal reference and input.
**Architecture:** Independent reference_anomaly array API + matcher-result loader + CLI/export. Robustly normalized color, structure and multiscale signals feed strict and displacement-tolerant branches.
**Tech Stack:** Existing Python, NumPy, OpenCV, unittest.
**Spec:** docs/superpowers/specs/2026-11-18-reference-anomaly-design.md

## Global Constraints
- No defect-color prior, XML boxes, ground truth, external model or new dependencies.
- Preserve strict evidence even where displacement tolerance suppresses it.
- Never score invalid reference as normal or as a defect; explicit masks distinguish it.
- No mutation of matcher inputs or existing artifacts.

## Review Focus
- Tiny and brightness-negative defects must survive.
- Hole boundaries must not create smoothed/gradient anomalies.
- Displacement search must use a single shift for all channels and signals.
- Broad changes must not disappear through score normalization.
- Export must not overwrite source files or accept nonfinite data.

### Task 1: Array analysis
Create reference_anomaly/core.py, __init__.py, tests/test_reference_anomaly.py.
Interface: AnomalyConfig, AnomalyResult, analyze_pair(input_image, reference, reference_valid, *, photometric_valid=None, source_map=None, config=None).
- [x] Write failing tests for bright/dark/color/structural anomalies, unchanged images, whole-image changes, shift tolerance, holes/seams and validation.
- [x] Implement masked multiscale/gradient signals, robust calibration, consistent shift search and dual hysteresis masks.
- [x] Run core tests and inspect signal strengths.

### Task 2: Files and CLI
Create reference_anomaly/io.py, __main__.py, README.md.
- [x] Write failing tests for loading actual matcher artifacts, export data precision, cleanup, malformed input and overwrite protection.
- [x] Implement result-directory loader, bounded candidate summaries, image/NPZ export, CLI radius option.
- [x] Run CLI on current SD-1 saved results, inspect montage and finish regression suite.

Execution: user said “来吧” after discussion; implement accepted approach on current feature branch, preserve prior uncommitted work. No commit/publish requested.

Decisions: default candidate high/low thresholds 1.8/1.0, exposed by CLI. Raw evidence remains unchanged by these thresholds. Heatmaps use fixed 0..3 gamma 1.4, with full float32 data in NPZ.
Fresh review found that a seam-only high seed could promote trusted weak pixels through priority hysteresis. Reproducing test failed, then passed after restricting priority seeding/growth to trusted pixels. Strict candidates preserved.
Focused verification: 13 tests passed; real SD-1 output generated and montage inspected. Final full-suite rerun: 185 tests passed (78.874 seconds). `git diff --check` passed. No commit/publish performed.
