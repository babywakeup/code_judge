# Periodic completion implementation plan

User requested integration of the previously demonstrated periodic overlay into
both matcher backends, including full valid-ROI label prediction.

Design: keep direct geometry intact; expose completed image/validity and label
provenance separately. Opt-in periodic_completion, enabled in SD-1 configs.
Infer a source lattice from repeated anchor-class components, falling back to
target image autocorrelation. Explicit source period vectors may override it.
Generate bounded integer shifts, refine and validate image agreement, then fill
only unknown pixels from actual valid source labels. Never infer known background
from missing annotation. Report complete/incomplete label coverage explicitly.
Preserve semantic/instance acceptance gates. No target annotations guide matching.

## Tasks
- [x] Add failing tests for border/corner completion, both backends, real public
  API transfer, annotation validity, wrong image evidence, no-period fallback,
  rejected/instance policy, file exports and stale artifact cleanup.
- [x] Implement periodic_completion.py and integrate configuration/result/API.
- [x] Add two-opacity self-contained preview and report/export updates.
- [x] Run full suite, run both SD-1 configs, inspect masks and previews.
- [x] Independent review and correction if needed.

No commit/branch changes requested. Work in current workspace.

## Verification record
- RED: 7 test methods initially failed because completion configuration was absent.
- Core API tests passed before file export tests; added export/cleanup integration.
- Full suite passed 40/40 before review.
- Independent review identified whole-overlap scoring could hide a corrupted
  missing border. Added localized corruption regression, observed RED, then
  validated every new connected region with bounded working-image context.
- Local regression GREEN; full suite now 41/41.
- Final real-image trials after local verification fix: both backends 100% ROI,
  321604 added pixels, zero unknown pixels. Checked actual exported PNG masks:
  valid ROI all 255; excluded footer all zero. Inspected completion region image.
