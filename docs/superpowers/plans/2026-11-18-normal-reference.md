# Normal Reference Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a photometrically corrected normal reference alongside existing segmentation outputs.
**Architecture:** Track exact accepted source fragments, robustly fit template-to-input gain/offset per fragment, compose and export reference with validity and provenance.
**Tech Stack:** Python, NumPy, OpenCV, unittest; existing dependencies only.
**Spec:** docs/superpowers/specs/2026-11-18-normal-reference-design.md

## Global Constraints
- Current input and normal templates only; no XML boxes or ground truth in fitting.
- Preserve segmentation and existing output directory conventions.
- Never paste target appearance into the normal reference.

## Review Focus
- Bright defects must remain absent from reference and must not bias gain/offset.
- Flat, saturated or insufficient samples must report fallback rather than confident fitting.
- Object-only matches and refined support fragments must retain exact accepted transforms.
- Rejected reruns must clear previous reference artifacts.
- Gray/BGR/BGRA images and excluded bottom ROI must remain valid.

### Task 1: Photometric reconstruction
Files: new `_internal/reference/step.py`, models, tests/test_normal_reference.py.
Interface: ReferencePiece(template_index, matrix, mask, kind); NormalReferenceStep.run(inputs, result).
- [x] Write and run failing tests for gain/offset with defects, flat/saturated fallbacks, channels and invalid pixels.
- [x] Implement bounded deterministic robust fitting, composition, source IDs and diagnostics.
- [x] Run focused tests to green.

### Task 2: Geometry integration
Files: pipeline, segmentation transfer/periodic, objects matching/completion/axes.
- [x] Add integration tests for direct, periodic, supplemental and object fragment provenance.
- [x] Capture actual fragment transforms and masks without changing segmentation decisions.
- [x] Run matcher, periodic and object regression tests.

### Task 3: Export and documentation
Files: io/export.py, README.md, ARCHITECTURE.md, tests/test_normal_reference.py.
- [x] Test reference artifacts, manifest, validity, stale cleanup and malformed arrays.
- [x] Export PNGs and source map to existing directory; document API and limitations.
- [x] Run complete unittest suite, inspect a real configured output if available, review diff.

Execution note: User explicitly requested implementation after the design discussion; proceed in place on feature branch without an additional approval round. Do not commit or publish unless requested.

Review: fresh reviewer reproduced displaced-object ghosting; regression written RED then exclusion fixed GREEN.
Additional fractional-ROI interpolation regression also RED then GREEN. Focused suite: 45 passed.
Independent review checked support transforms, y-axis adaptation, provenance and stale cleanup.
Final verification: `python -m unittest discover -s tests` passed all 172 tests.
Real SD-1 config regenerated existing `periodic_template_matcher/results/sd1_opencv` outputs;
reference and masks visually inspected. `git diff --check` passed. No commit or publish performed.
