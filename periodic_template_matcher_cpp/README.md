# Native periodic template matcher: build, fixtures and debugging

The C++ core performs matching/completion/evaluation. Python remains responsible
for file decoding, LabelMe annotations, workflow configuration and result IO.
This is **not a standalone image-file CLI**, and complete Python/C++ parity is
**not established**. Current Python OpenCV and the local C++ SDK are both 4.14.0.
See [verified status and limitations](docs/status.md).

## Build on this Windows checkout

Use the repository's `.venv` Python 3.12, not an unrelated PATH interpreter.
From the repository root, run `.venv/Scripts/python.exe
periodic_template_matcher_cpp/build.py --config Debug` or `.venv/Scripts/python.exe
periodic_template_matcher_cpp/build.py --config Release --jobs 2`.
These are separate Ninja single-configuration build trees. Passing only
`cmake --build ... --config Release` to the Debug tree does not change its build type.

[build.py](build.py) locates CMake/Ninja beside the selected Python interpreter,
imports pybind11 from that environment, finds the deepest local OpenCV SDK config,
and uses the existing Zig wrappers under `.deps/tools` when MSVC is unavailable.
For a different SDK, pass `--opencv-dir` pointing to the directory containing the
actual OpenCVConfig.cmake, rather than a Windows dispatcher that rejects Zig.
`--configure-only` configures without compiling; `--jobs N` limits parallelism.

Build-time dependencies: Python development headers/import library, NumPy,
pybind11, CMake, Ninja, a C++17 compiler, the OpenCV core/imgproc/photo/video SDK,
and the local nlohmann/json and pocketfft headers. Existing
[prepare_native.py](tools/prepare_native.py) can prepare the local SDK, but it
downloads and builds dependencies; it is not necessary on this prepared machine.
The wrappers may reference a Zig executable outside this package. Moving the
checkout requires inspecting/regenerating those wrappers and reconfiguring,
not merely copying the CMake cache.

Outputs are `build/debug/ptm_debug.exe` and `build/release/ptm_debug.exe`, with symbols alongside them. Each configuration stores its extension in its own `build/<configuration>/python/` directory. `build.py` copies the chosen extension into the package and records the active configuration and SHA-256 in `build/active-native.json`. Restart Python processes using the extension before switching builds.

## Export a real fixture

[export_fixture.py](tools/export_fixture.py) performs IO and
`_native.prepare_inputs`, not matching. From the repository root:

- Synthetic identity smoke: `.venv/Scripts/python.exe
  periodic_template_matcher_cpp/tools/export_fixture.py --smoke --out
  periodic_template_matcher_cpp/build/debug/fixtures/smoke`.
- Add identity references: use `--smoke --smoke-objects` with a different output
  directory. This exercises A/B/C masks and object metadata.
- Actual SD-1 input: `.venv/Scripts/python.exe
  periodic_template_matcher_cpp/tools/export_fixture.py --config
  periodic_template_matcher/config.sd1.labelme.opencv.json --out
  periodic_template_matcher_cpp/build/debug/fixtures/sd1`.

Output is a fixture.json manifest plus numbered raw arrays. Choose a **new**
directory for each export: existing manifests/raw files are not overwritten.
On failure, a partial output directory may remain; discard it or choose a new one.
All configured image/mask/LabelMe paths resolve relative to the config, not cwd.
Full-size normalized inputs are exported; workflow `max_side` is applied later
inside matching, not by this exporter.

The current [workflow](../periodic_template_matcher/_internal/io/workflow.py)
has no `load_job` API. The exporter reuses its `unique_templates`, `layout_axis`,
`object_layout`, `read_image`, `load_labels`, and `load_objects` functions,
mirroring the IO-only prefix of `run_config`. It preserves config expansion,
duplicate conflict checks/aliases, valid heights, ROI/annotation separation,
class mapping and partial identity layouts. It does not invoke `run_config`,
`match`, or `save_result`. A future `load_job` is preferred only if it returns
the explicitly checked `(MatcherConfig, match_keyword_dict, metadata_dict)`
contract; that future branch is not verified against a current workflow API.

Fixture inputs have exactly the keys consumed by
[debug/main.cpp](debug/main.cpp): `target`, `target_valid`, `regions`, `templates`,
`labels`, `valids`, `annotations`, and `objects`. Every non-null array is
`{rows, cols, type, file}`; files are relative to the manifest directory, row-major,
contiguous, little-endian, and retain BGR/BGRA channel ordering. OpenCV type is
`depth + ((channels - 1) << 3)`; native-normalized labels are uint16 and validity/
object masks uint8. Missing annotations/labels remain JSON null. References keep
`identity`, `mask`, `background_known`, `shape_index`. Metadata lists byte sizes,
shapes, Python OpenCV version and input provenance; it is ignored by the C++ reader.
This raw format targets the current little-endian executable; it is not a portable
big-endian interchange format. Only load trusted fixtures: the existing C++ reader
does not comprehensively sandbox filenames or reject oversized allocations.

Run `periodic_template_matcher_cpp/build/debug/ptm_debug.exe` with the exported
manifest path as its sole argument. No arguments selects the built-in smoke.
With a fixture, exit 0 means the program completed, **not** that registration was
accepted; inspect the printed `accepted` and `status` fields.

## Verify a source breakpoint (without installing an extension)

Run `.venv/Scripts/python.exe periodic_template_matcher_cpp/tools/verify_debugger.py
--fixture periodic_template_matcher_cpp/build/debug/fixtures/smoke/fixture.json`.
It defaults to the executable built by Debug and discovers the line containing
the `ptm::match` call in [main](debug/main.cpp#L39). `--log` saves the full debugger
transcript to an existing parent directory. Default timeout is 120 seconds; use
`--timeout 300` for slower real inputs. `--source` plus `--line` can select another
executable source location, but that line must really be reached by the fixture.

The default `auto` backend first uses the portable
`.deps/codelldb/extension/lldb/bin/lldb.exe` CLI. It checks LLDB's actual stop reason,
frame source/line and successful continuation/exit, not just breakpoint creation.
If CLI is absent/fails, it uses `adapter/codelldb.exe` with `lldb/bin/liblldb.dll`
over loopback reverse DAP TCP. Use `--backend cli` or `--backend dap` to verify
each independently; `--extension` accepts a different extracted extension root.
The DAP path checks the stop event, top stack frame and exit event, and captures
locals. A `verified`/pending DAP breakpoint alone is **not evidence of a hit**.
No VS Code extension installation is needed for these scripts. Existing
[prepare_debugger.py](tools/prepare_debugger.py) downloads the portable debugger
if it is missing. Exit 1 and `verified: false` enumerate blockers.

## VS Code and source mapping

Open [ptm.code-workspace](ptm.code-workspace). Its named `repo` and `ptm` folders
give relative paths for the `.venv`, Debug/Release build tasks and launch settings.
Use “PTM: built-in smoke” or “PTM: exported fixture” (prompts for a manifest path).
Both build Debug before launching. Set a source breakpoint at the `ptm::match`
call or inside [the pipeline](src/pipeline.cpp). F5 uses `type: lldb`, which
**does require** the CodeLLDB VS Code extension; the CLI/DAP verifier above does not.
VS Code uses the LLDB library bundled with the installed CodeLLDB extension. The
portable LLDB library under `.deps` is used by the standalone verification script.

The existing debug symbols refer to `D:/codes/code_judge`. Workspace `sourceMap`
maps that build-time prefix to `${workspaceFolder:repo}`. Keep executable and
matching symbols together, including the PDB when present. If built elsewhere,
replace/add the **actual compiled prefix**; do not map to the fixture directory.
For the verifier use `--source-map D:/codes/code_judge E:/new/checkout` (repeat for
multiple prefixes). CLI uses `target.source-map`; DAP uses `sourceMap`. Source
mapping fixes source lookup, not missing symbols or stale binaries. Rebuild after
source edits; an optimized Release executable is not a substitute for Debug.

## Deployment and Python IO/API

- Native-only fixture deployment: executable, its matching symbols for debugging,
  manifest and all referenced raw files. The local OpenCV SDK is static. Python,
  pybind11 and CodeLLDB are not needed to run the resulting executable normally.
  Do not assume a different/shared SDK has no OpenCV/compiler runtime DLL needs;
  validate dependencies and launch on the destination Windows machine. The
  local run is not proof of a clean-machine deployment.
- Python API deployment: the matching-ABI `_native` extension, both
  `periodic_template_matcher_cpp` and `periodic_template_matcher` Python packages,
  NumPy and Python OpenCV plus the IO dependencies in
  [the Python requirements](../periodic_template_matcher/requirements.txt).
  The current binding is CPython 3.12 Windows x64, not ABI-independent. OpenCV's
  Python wheel is separate from the C++ SDK; don't mix their DLL directories in PATH.
  SDK/header/CMake/compiler packages are build dependencies, not necessary Python
  runtime packages when deploying a working extension.
- [api.py](api.py) exposes `PeriodicTemplateMatcher.match`, `evaluate`, `save_result`
  and `run_config`. Configure using Python `MatcherConfig`; pass uint8 gray/BGR/
  BGRA NumPy images at least 32 by 32, one label entry per template (uint16 class-ID
  mask, binary-class-mask list, or None), source-sized ROI/annotation/object masks,
  optional valid heights and configured ordered identity references. Native
  `prepare_inputs` validates and normalizes the boundary; it does not match.
  `evaluate` keeps ground truth out of matching. `save_result` is Python export IO.
- Existing run workflow CLI: `.venv/Scripts/python.exe -m
  periodic_template_matcher_cpp --config
  periodic_template_matcher/config.sd1.labelme.opencv.json --out <new-result-directory>`.
  The adapter substitutes the native matcher in a per-call copy of workflow
  globals; original shared Python workflow state is not patched.

## Validation scope

Debug/Release builds, smoke/object/SD-1 fixture execution and **actual source
breakpoint hits** are recorded in [docs/status.md](docs/status.md). The current native unittest discovery passed 48 tests under the documented numerical comparison contract. Neither successful execution nor debugger
verification establishes whole-pipeline numerical/diagnostic parity. OpenCV version
differences may contribute to numerical differences, but are not a demonstrated
explanation of every failure.
