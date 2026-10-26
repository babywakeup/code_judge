# Inference readability and shared preprocessing

Goal: simplify the existing inference pipeline, annotate every package function's inputs and outputs, and preserve public APIs, coordinates, scores and config compatibility.

Architecture: prepare each image once per multi-template search; reuse target features and lattice diagnostics. Keep boundary validation, use private functions for prepared matching, and project ROI masks directly. Describe NumPy data contracts with shared type aliases rather than repeated array unions.

Execution: continue in the user's existing workspace to include the uncommitted scale/grouped-config changes being debugged. Do not commit or discard existing changes.

- [x] Add regression coverage for shared target preprocessing, equivalence to standalone registration, and mask projection.
- [x] Refactor registration into image preparation, coarse candidate selection, refinement and ranking.
- [x] Remove unused gray conversions, duplicate workflow ROI processing and zero-label warps.
- [x] Annotate all functions under periodic_template_matcher, including CLI/report helpers, and document array shapes and pipeline responsibilities.
- [x] Run the full unittest suite, check annotations/imports and review the final diff.

Validation: baseline 53 tests passed; final `python -m unittest discover -s tests -v` passed 56 tests. All 68 explicit function signatures have parameter/return annotations and runtime type hints resolve. All three SD1 configurations parse. UTF-8 text and `git diff --check` passed.

Read-only independent review found no important inference regressions. Its IO dtype and Sequence-contract findings were addressed: image IO uses a general array type for uint16 label support, and matcher sequence checks agree with their signatures.

Review focus: mixed template sizes and shared scale; blank images and rejected matches; partial annotation validity; explicit heights vs bottom exclusions; periodic completion and unchanged original-coordinate matrices. No new runtime dependencies.
