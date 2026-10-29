# Ordered objects implementation plan

Goal: implement approved class-aware-object-matching design, preserving existing
single-class behavior when the new objects option is disabled.
Spec: ../specs/2026-10-28-class-aware-object-matching-design.md

Architecture: retain original polygon masks and configured identity order in
annotation loading; an independent ordered_objects module builds appearance
candidates with overlap-normalized correlation, then selects coherent cyclic
orders. Existing geometry and per-pixel annotation validity remain authoritative.
Class-1 periodic completion cannot transfer class 2 or infer its background.
Semantic output remains class 2; identity map and evidence are separate artifacts.

Tasks:
- [x] RED tests for ABC/BCA/CAB and clipped borders, corrupted/ambiguous objects,
  invalid annotation/config, and file persistence.
- [x] Add ordered reference parsing/configuration and array API plumbing.
- [x] Implement masked object correlation, order resolution, partial visibility,
  diagnostics and original-coordinate identity masks.
- [x] Merge class-specific outputs and prevent periodic class-2 transfer.
- [x] Extend saved result and independent viewer with identity metadata.
- [x] Run full tests and both actual input examples; inspect output overlays.
- [x] Independent review; regress and fix important findings.

Implementation in current workspace, no commit or branch changes requested.

Verification record:
- Initial ordered-object tests failed due to missing configuration/modules.
- ABC/BCA/CAB, left/right partial views, noise, indistinguishable identities,
  source clipping, independent viewer/rendering and JSON/export cleanup covered.
- Independent reviewer found clipped source could falsely certify background and
  instance mode could silently delete class 2. Both reproduced RED, fixed GREEN.
- Added regression for tiny vertical fragments outranking whole-object evidence;
  verified RED then GREEN with separate vertical/horizontal visibility penalties.
- Full suite: 77 tests passing. Geometric rejection remains authoritative; only
  valid geometry plus confident ordered identities can resolve a template tie.
- Actual 1379453: ABC, group margin .02605, final ROI known coverage 100%.
- Actual 1383316_defect: BCA, group margin .02833, coverage 94.06%; uncertain
  C extent remains unknown. Actual identity overlays visually inspected.
- Scene-specific object margin configured .02, API default remains .03; neither
  score nor coverage is a measured identity/segmentation accuracy.
