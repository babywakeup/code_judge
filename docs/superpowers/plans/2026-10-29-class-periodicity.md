# Per-class translation-symmetry policies

## User semantics

Along an axis, "period=1" means no nonzero translation aligns the underlying
structure; "period>1" means a repeating translation exists. These are not
occurrence counts or period lengths in pixels. White defects do not change the
healthy structure's declared symmetry or remove its transferred segmentation.

## Configuration and scope

Added options.periodic.class_periodicity, mapping foreground class IDs to allowed
source-image axes: [], ["x"], ["y"], or ["x", "y"]. Keep actual pixel distances in
the existing shared period_vectors/anchor/image-autocorrelation pipeline.

- Missing field / empty mapping preserves legacy normal-class xy and ordered
  identity-class y behavior.
- A nonempty mapping is explicit: omitted classes cannot repeat.
- Canonical string or integer IDs 1..65535 are accepted, with duplicate numeric
  identities, unknown/duplicate axes and invalid types rejected.
- Ordinary semantic classes support all four modes. The existing cyclic identity
  locator assumes one horizontal occurrence per identity; ordered classes only
  support [] or y and reject x instead of silently ignoring it.
- Current SD-1 config explicitly declares class 1 xy and class 2 y; preserve the
  user's current 1352308_defect input and all registration thresholds.

## Algorithm safeguards

Direct transfer is unchanged. For each periodic candidate, check every active
lattice generator against each class's axes (single-axis skew tolerance 15%).
Do not let sums of forbidden oblique generators bypass the policy.

Copy only allowed positive labels. Do not recode forbidden labels as background.
Periodic background supplies negative knowledge only when every base foreground
class supports the candidate shift; otherwise keep background unknown. Exclude
the independently localized ordered class from base background policy because
its own validity is merged separately, preserving the existing base pipeline.

Separate globally verified image geometry from local label-copy acceptance in
diagnostics. Geometry may still guide independent object support when all base
classes prohibit copying or base labels already cover the ROI. Object support
retains its own direct-pixel, structure, edge, annotation and white-occlusion
checks, and now also checks the configured identity-class axes.

## Verification

- Observed RED before adding the new configuration and methods.
- 12 new tests cover four modes, cross-class x/y rules, forbidden foreground and
  background, omitted IDs, legacy pixel equivalence, validation, diagonal basis,
  nonperiodic object support, white defects, and geometry independence from base
  label permission/coverage. Full suite: 105 tests pass.
- Compared explicit SD-1 rules with legacy defaults on actual inputs 1352308
  defect, 1383316 defect, 1343428 and 1379453. All accepted, with orders CAB,
  BCA, ABC, ABC respectively and zero unknown ROI pixels. Final labels, validity,
  base labels/validity and object identities are pixel-exact across both modes.
- Re-generated results/sd1_opencv for the unchanged current 1352308_defect input;
  checked saved config, acceptance, CAB order and zero unknown pixels.
- Editor diagnostics clear for changed code/config. These checks establish
  policy routing and regression preservation, not segmentation-GT accuracy.