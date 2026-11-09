# Matcher Modularity Implementation Plan

**Goal:** Improve readability and responsibility boundaries without changing the public API or numerical behavior.

**Architecture:** Keep the existing domain directories and explicit pipeline. Extract reusable registration kernels and object layout/search helpers; retain existing import entry points and algorithm injection points.

**Tech Stack:** Python, NumPy, OpenCV, unittest.

## Design and constraints

The user selected the compatibility-preserving approach. Algorithm tuning and performance-policy changes are outside this refactor. Existing staged and unstaged work must remain intact; work in the current checkout because it contains the implementation being refactored. Do not commit unrelated changes.

Alternatives considered: retain the current large algorithm modules (less movement, weaker separation), or tune algorithms while splitting modules (larger behavioral risk). Extracting cohesive kernels keeps the numerical formulas and execution order intact.

## Tasks

- [x] Establish baseline with `.venv/Scripts/python.exe -m unittest discover -s tests -q`.
- [x] Extract registration `correlation.py` (FFT peaks and lattice diagnostics) and `scoring.py` (NCC, scoring and ECC); keep core imports available for compatibility. Keep feature extraction in core to preserve existing injection tests.
- [x] Extract object `layout.py` (quality, spacing and cyclic paths) and `appearance.py` (masked correlation peaks); annotate matching inputs and prediction arrays. Preserve helper imports from matching.
- [x] Clarify step-local evidence-mask construction and matrix selection without changing acceptance gates.
- [x] Update architecture documentation; run the full suite and inspect extraction equivalence and imports.

## Review focus

Preserve low-texture rejection, shared target preprocessing, FFT/OpenCV backend selection, cyclic and vertical layouts, partial-reference behavior, template ties, and verified support completion. Existing numerical regression and pipeline tests exercise these behaviors. Compare extracted function ASTs to their original bodies to catch accidental formula edits.

## Verification record

- Baseline: 137 tests, 136 passing. Existing `test_actual_sd1_templates_and_input_quarter_turns` fails for clockwise class 2: IoU 0.8966673810179846, threshold 0.90.
- After refactor: the same 137 tests, the same single failure and identical IoU; no new failures.
- Focused package, pipeline and backend checks: 28/28 passing.
- Extraction check: 21 original function/class bodies match their AST snapshots, ignoring type annotations; prediction field annotations were intentionally clarified.
- `git diff --check`: passed (existing CRLF conversion warnings only).
- Algorithm tuning is deferred as agreed; changing rotation behavior would violate the compatibility boundary of this task.
- Independent review: no critical or important issues. Updated older reading-guide links to agree with the extracted module locations.
