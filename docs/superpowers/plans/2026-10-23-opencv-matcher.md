# OpenCV Matcher Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Select FFT or OpenCV coarse matching through options.backend.
**Architecture:** Replace translation correlation only; share refinement and exports.
**Tech Stack:** Python, NumPy, OpenCV, unittest.
**Spec:** ../specs/2026-10-23-opencv-matcher-design.md (approved by user).

## Global Constraints
- Default backend remains fft; accepted values are fft and opencv.
- Padding cannot count as valid overlap; preserve signed partial-overlap search.
- Preserve shared scoring, ECC and label transfer.

## Review Focus
- Equal-size signed shifts and unequal dimensions must retain coordinate direction.
- Invalid target pixels must not influence matching.
- Zero feature energy must not produce accepted peaks.
- Unknown backend names must fail before running.
- JSON backend selection must reach registration and result metadata.

## Task 1: Backend and regression tests
- [x] Add tests/test_opencv_backend.py: signed shifts, differing sizes, mask holes,
  blank features, FFT agreement, public label transfer and JSON selection.
- [x] Run tests and observe failure before implementation.
- [x] Add backend validation to config.py; dispatch from registration.py.
- [x] Create _internal/opencv_registration.py with translation_peaks(source,
  target, sm, tm, cfg), returning (merit, dx, dy) tuples. Use padded TM_CCORR
  for dot products, energies and overlap; share eligibility and peak selection.
- [x] Run new tests and existing unittest suite.

## Task 2: Trial and documentation
- [x] Add explicit fft setting and separate OpenCV SD-1 config.
- [x] Document backend selection, limits and commands in README.md.
- [x] Run SD-1 OpenCV trial into a new directory; verify exported labels.
- [x] Review changes and record verification outcomes below.

## Execution record
Implementation in current workspace as requested; no commits or branch changes.

- Task 1: new tests failed first (5 missing-feature errors), then passed (5/5).
- Task 2: full suite passed (33/33). Fixed an existing Windows-only test
  encoding assumption by reading the UTF-8 report explicitly as UTF-8.
- SD-1 OpenCV trial: accepted template 1258237, score 80.6141121373382;
  labels exported, full-image label coverage 0.8562640545252413.
- Result metadata confirmed backend=opencv; overview visually checked.
- Independent code review: no actionable correctness or regression issues. Reviewer could not run Python in its sandbox; full test evidence above comes from the primary execution.

