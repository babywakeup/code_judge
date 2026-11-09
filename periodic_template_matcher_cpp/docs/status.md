# Verification status — 2026-11-06

## 2026-11-07 continuation

The earlier status below describes a previous session and is retained as history. Current SDK and Python OpenCV are both 4.14.0. The current build script publishes Debug/Release from separate output directories and records the selected artifact in `build/active-native.json`. A Debug source breakpoint was hit after source formatting; native CTest passed 2/2, the original Python suite passed 161/161, and the current native differential suite passed 48/48 under the documented numerical comparison contract.

Exact real SD-1 comparisons from `build/parity-resume-debug.json` completed on the base image and both quarter turns. The status, acceptance, ranking, and selected template agree, but arrays differ: mapped-label pixels differ by 818 / 231 / 319 and identity pixels by 742 / 117 / 253 respectively. These results do not establish exact functional parity. The report predates the final input validation guards and source formatting, which do not change valid matching arithmetic; it identifies its tested binary by SHA-256. Debug timing is not a Release performance benchmark.

The synthetic integration checks compare rejected-geometry projections against each implementation's own returned matrix, and compare supplemental images against each implementation's own accepted matrix; labels, identity maps and masks remain exact cross-language assertions. This change reflects independently observed ECC build sensitivity and does not lower matching thresholds. The raw SD-1 report retains exact array comparisons.

## Scope

Only five source/documentation files were created for this task:

- [tools/export_fixture.py](../tools/export_fixture.py)
- [tools/verify_debugger.py](../tools/verify_debugger.py)
- [README.md](../README.md)
- [docs/status.md](status.md)
- [ptm.code-workspace](../ptm.code-workspace)

No existing CMake, C++ core, bindings, Python workflow, tests or other source files
were edited. Builds generated their normal artifacts; exported fixtures and
debugger logs were placed in the Windows temporary directory, not checked in.

## Observed environment and builds

- Configured interpreter first: repository `.venv/Scripts/python.exe`, CPython
  3.12.13, Windows x64. Runtime check: Python `cv2.__version__ == '4.14.0'`.
- Debug: existing VS Code task ran the equivalent of `.venv/Scripts/python.exe
  periodic_template_matcher_cpp/build.py --config Debug`, exit 0.
- Release: same interpreter ran [build.py](../build.py) with `--config Release
  --jobs 2`, exit 0; Clang 22.1.8 via `.deps/tools/cxx.cmd`, OpenCV 4.13.0,
  pybind11 3.1.0; executable and extension linked successfully.
- Existing signed/unsigned comparison warnings in the C++ core were not changed.
- Release writes the same package-level native extension as Debug. The final
  extension is Release in this session; executable build trees remain separate.
  Clean generated Debug outputs and build Debug again to force a binding switch;
  a newer shared output can otherwise cause Ninja to report no work to do.

## Fixtures actually exported and executed

| Fixture | Raw arrays | Templates | Object references | Release execution |
| --- | ---: | ---: | --- | --- |
| Synthetic BGR smoke, seed 42 | 5 | 1 | null | exit 0, accepted=true, status=accepted |
| Synthetic A/B/C smoke | 8 | 1 | 3 | exit 0, accepted=true, status=accepted |
| Actual SD-1 LabelMe/OpenCV run config | 14 | 2 | 3 + 3 | exit 0, accepted=true, status=accepted |

Actual config: [config.sd1.labelme.opencv.json](../../periodic_template_matcher/config.sd1.labelme.opencv.json).
Sources were read from its configured SD-1 images and templates2 LabelMe files.
Normalized object windows are A/B/C and D/A/B, full cycle A/B/C/D. Configured
valid height is 1891. All raw file lengths were checked against metadata byte
counts. Synthetic ROI height 90 gives 10,080 valid pixels on a 96 by 112 image.
The exported null annotations remain null. No image matching occurs in export.

## Source breakpoint evidence (not merely symbol discovery)

Portable CLI reports `lldb version 22.1.8-codelldb`. Both backends were run:

- **CLI, exported smoke**: actual stop reason 3 equals
  `lldb.eStopReasonBreakpoint`; top frame `main`, source
  [debug/main.cpp](../debug/main.cpp#L39), line 39; `argc=2`,
  `inputs` variable valid. Continue: state 10 equals `lldb.eStateExited`, exit 0.
- **CodeLLDB DAP TCP, exported smoke**: breakpoint id 1; actual `stopped` event
  reason `breakpoint`, `hitBreakpointIds=[1]`; top frame `main` at the same source
  line 39, `argc` local value 2, `inputs`/`cfg` scopes readable; `exited.exitCode=0`.
  The initial DAP response said `verified=true`, `Resolved locations: 0`;
  that response was **not** used as proof. The subsequent hit event and source
  frame prove execution reached the source breakpoint.
- **CLI, actual SD-1 fixture, after Debug rebuild**: same breakpoint source/line,
  stop reason 3, `argc=2`, inputs valid; continue and exit 0.
- **CLI, object smoke, after Debug rebuild**: same breakpoint source/line,
  stop reason 3; continue and exit 0.

After the final verifier edits, both CLI and DAP were run again with an explicit
`D:/codes/code_judge` to current-checkout source map: source line 39 was hit and
the process exited 0 in both cases. This tests the current-prefix map, not a
physically relocated checkout. The CLI comparer handles LLDB line entries that
still retain their compile-time prefix using the configured longest-prefix map.

Artifacts for this session are under Windows `%TEMP%`, directory
`ptm-fixture-proof-e0b28c7449904b94b61fda0ad8793dbc`: smoke/objects/sd1 fixture
subdirectories, cli.log, dap.json, sd1-cli.log, objects-cli.log, cli-mapped.log,
dap-mapped.log. These are transient
and may be removed by Windows cleanup. Use the scripts to regenerate evidence.
Workspace launch definitions are supplied, but interactive F5 was not used for
this verification; CLI and real DAP TCP debugging were used instead.

Final checks: both Python files passed AST parsing and Pylance syntax checks;
workspace JSON parsed and contains two launch configurations; editor diagnostics
reported no errors in the five new files. Attempting to export over an existing
fixture returned exit 1 with an explicit refusal, and a missing debug executable
returned exit 1 with `verified: false` and its missing path as the blocker.

## Regression/parity blockers — not green

Release native binding unittest discovery was actually run with `.venv/Scripts/
python.exe -m unittest discover -s periodic_template_matcher_cpp/tests -v`.
It ran **36 tests**; result **FAILED (failures=5, errors=18)**, exit 1.
Counts include failed subtests; this is unittest discovery, not a claim that every
possible pytest/function test in the repository was collected.

- 18 errors: current native module lacks `correlate`, used by dtype/normalization
  subtests in [test_cpp_registration.py](../tests/test_cpp_registration.py).
- Two object-only integration subtests (FFT/OpenCV) have transform differences;
  observed y-translation is 0.0 native versus approximately 18.80697 Python.
- Two supplemental-completion subtests differ in `completed_template` pixels.
- One periodic-lattice test differs in candidate vector selection/order.

These concern existing core/binding/parity tests outside the allowed edit scope;
no attempted core or test fixes were made. Python OpenCV 4.14.0 versus C++ 4.13.0
is a known mismatch, **not proof that version skew causes every observed failure**.
In particular a missing binding is not explained by OpenCV version skew.
Successful fixture replay/acceptance and source debugging do **not** prove all
parity or certify clean-machine deployment. Full parity remains blocked pending
binding and numerical/diagnostic investigations, ideally against matched OpenCV
versions and explicit tolerances.

## Boundaries and remaining limitations

- No current `load_job` exists. Export reuses the real workflow IO helpers;
  the guarded future tuple-contract loader branch is not validated against an
  actual `load_job` implementation.
- Raw fixtures are trusted, little-endian snapshots of native-normalized inputs;
  the existing C++ raw reader's security/validation behavior was not modified.
- Executable continuation exit 0 is checked; fixture-mode acceptance must be
  inspected separately (it was inspected for the three Release fixtures above).
- CLI and DAP have no source-breakpoint blocker on this prepared machine.
  Relocated-source mapping, another machine's deployment DLL set, and interactive
  VS Code extension launch have not been independently certified.
