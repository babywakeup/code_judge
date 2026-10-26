# Two-stage prediction inspection

Goal: replace matcher.visualize with matcher.save_result, then inspect saved input and predicted labels using a desktop app under periodic_template_matcher/app.

Stage 1: a shared result exporter writes lossless original-sized input, uint16 class IDs, label validity, optional periodic validity, optional ground truth/evaluation and a JSON manifest. Workflow delegates to this exporter instead of building HTML/contact sheets. Rejected and unannotated results never become empty valid predictions. Remove old visualize and HTML report implementation.

Stage 2: a Tkinter app (no extra pip dependency) loads the manifest and files. A separate testable renderer supports original, mask, overlay and synchronized side-by-side modes; class/source filtering; unknown regions; pixel inspection; zoom/pan. File opening/reload never reruns inference.

- [x] Regression tests for result round trips, uint16 IDs, stale files, missing predictions and viewport rendering.
- [x] Implement exporter and replace matcher/workflow/CLI visualization path.
- [x] Implement typed result loader, renderer and Tkinter application with module entry point.
- [x] Update run.sh, README and old visualization tests for the two-stage interface.
- [x] Full unit tests, actual SD1 export and desktop GUI smoke verification.

Keep current configuration formats and original-coordinate inference behavior. Use the current workspace including pending class_map fixes; do not commit user work. Save existing artifact names for class IDs/validity and clean stale legacy generated files in the chosen output directory. The app has no browser hosting or external service dependencies.

Verification: 62 unittest cases passed. Actual SD1 OpenCV inference accepted template 0 with score 80.61 and wrote the new manifest/data. Desktop smoke covered all view modes, class/source filters, zoom, hover, reload and rejected results. All 83 functions have annotations and runtime type hints resolve. CLI help, bash syntax and Tk-independent stage-1 import passed.

Independent review findings fixed with regression cases: hover sampling now matches rendered nearest labels exactly; input-file protection includes deduplicated template paths. The palette uses all 16 class-ID bits to avoid automatic 256-ID color repetition. Existing fake demo predictions and HTML/contact-sheet generation were intentionally removed in favor of persisted real prediction data and the viewer.
