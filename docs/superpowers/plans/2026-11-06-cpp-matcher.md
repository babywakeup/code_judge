# C++ Matcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the complete current matcher algorithm as a debuggable C++ core called from Python.
**Architecture:** Standalone C++17/OpenCV core; pybind11 array boundary; Python IO and result types reused. Independent algorithm subsystems share a frozen header before parallel implementation.
**Tech Stack:** C++17, OpenCV, CMake, pybind11, nlohmann/json, native debugger.
**Spec:** `docs/superpowers/specs/2026-11-06-cpp-matcher-design.md` (approved by user: 按设计实现).

## Global Constraints
- Preserve all algorithm branches and existing Python source; no Python algorithm fallback from C++.
- Preserve original-coordinate transforms, unknown/background distinction and discrete identity semantics.
- Float comparisons initially atol=1e-4/rtol=1e-5; original-point displacement target 0.1px.
- Debug means an actual source breakpoint verified, not just compilation.
- Build products/dependencies ignored; do not commit unrelated staged user changes.

## Review Focus
- Noncontinuous/temporary NumPy arrays must not produce dangling pointers (Task 1/5).
- Rejected registration may still yield objects-only predictions (Task 4/5).
- Stable tied scores and NumPy rounding/percentiles must preserve decisions (Task 1/2).
- Partial references and rotated layouts must preserve unknown regions (Task 3).
- OpenCV version/build differences must be reported, never hidden by altered thresholds (Task 5).

### Task 1: Build, shared types, numerical utilities and input boundary
Files: new `periodic_template_matcher_cpp/{CMakeLists.txt,CMakePresets.json,include/ptm/core.hpp,src/common.cpp,bindings/module.cpp,tests/test_cpp_core.py}`.
Interfaces: `Config=Json` normalized runtime values; `Features`, `Inputs`, `MatchResult`, `ObjectRef`, `ObjectPrediction`; functions declared in `include/ptm/core.hpp`.
- [ ] Write import/native warp/round/percentile/ownership tests and observe missing feature failure.
- [ ] Prepare compatible native toolchain and OpenCV C++ SDK, build core and binding.
- [ ] Implement shared numerical utilities, array boundary; run corresponding tests.

### Task 2: Registration
Files: new `src/registration.cpp`, `tests/test_cpp_registration.py`.
Interfaces: `features`, `score`, `refine`, `correlate`, `translation_peaks`, `periods`, `select_templates` in shared header.
- [ ] Write parity tests for known signed translation, ROI, FFT/OpenCV, score, ECC, ties and texture rejection; observe failures before implementation.
- [ ] Port Python registration modules preserving dtypes/ordering; verify tests.

### Task 3: Ordered objects
Files: new `src/objects.cpp`, `tests/test_cpp_objects.py`.
Interfaces: `match_ordered_objects`, `complete_object_support` in shared header, consuming registration numerical utilities.
- [ ] Write cyclic/partial/multi-reference/x-y-layout and defect-support parity tests; observe failures.
- [ ] Port appearance/layout/matching/axes/completion; verify tests.

### Task 4: Periodic semantics and evaluation
Files: new `src/periodic.cpp`, `src/evaluation.cpp`, `tests/test_cpp_periodic.py`.
Interfaces: `complete_periodic`, `evaluate_segmentation` in shared header.
- [ ] Write period, boundary foreground, per-class policy, rejection and evaluation-count parity tests; observe failures.
- [ ] Port periodic completion and evaluation; verify tests.

### Task 5: Pipeline, Python facade, debug runner and integration
Files: new `src/pipeline.cpp`, `python` facade, `debug/main.cpp`, fixture exporter, docs, standalone VS Code workspace config; extend binding/test surface.
Interfaces: `match(Inputs,Config) -> MatchResult`; Python facade same array API and result containers; standalone runner uses same core.
- [ ] Write full API parity tests (both backends, ROI, objects-only, multisource, rotation, arrays unchanged); observe failures.
- [ ] Implement preparation, seven-step orchestration, supplemental sources, fusion, facade and IO adaptation.
- [ ] Build Debug and Release, verify native source breakpoint and Python native call.
- [ ] Run original 161 tests and new parity suite; record all differences and real SD-1 runs.
- [ ] Review whole change, address important findings, document exact build/deploy/debug commands and status.

## Execution rulings / progress ledger
- User explicitly requested implementation now; continue without another plan approval round.
- Work in new `periodic_template_matcher_cpp` directory in the current checkout: user approved that location and the migration baseline includes uncommitted source unavailable in a HEAD-only worktree. Preserve staging and do not auto-commit.
- Shared config uses normalized JSON values with centralized validation/access rather than duplicating every field across binding code. Core remains independent of Python.
- Parallel workers own separate translation units/tests; shared header/build/binding owned by coordinator. Reviewer at integration checks all interfaces.
- Baseline: 161 tests passed; 63 files hashed; Python cv2 4.14.0, requirements pin differs.
