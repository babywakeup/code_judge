# Vertical instance layouts verified by quarter-turn tests

## Requirement and observed failure

Rotating templates, input and annotations together by 90 degrees turns a
horizontal identity layout into a vertical layout. The previous explicit
order_axis="y" rejection was an implementation limitation, not a geometric
limitation. The user's last run.sh traceback stopped at that rejection.

## Implementation

- Add runtime object_order_axis x/y, inferred from template instance_layouts
  before validating repeat_axes. Require the same axis across templates.
- Sort original LabelMe instances on x or y according to the declared layout.
- Reuse the common horizontal solver through reversible transpose adapters:
  image/masks transpose; source-to-target matrices conjugate as S H S, where
  S swaps x/y. Convert predicted arrays and diagnostics back to actual input
  coordinates, including centers, axis fractions, neighbor names, support and
  corrected transforms and target period vectors.
- Support completion repeats perpendicular to identity order: x order -> y
  support, y order -> x support. Preserve existing per-region image checks,
  white-defect evidence separation, ROI/annotation constraints and diagnostics
  dictionary references. Never duplicate identities along the order axis.
- Keep current unrotated source/config untouched; no in-place source rotation.

## Rotation conventions

For original left-to-right CAB and cycle ABC:
- Clockwise 90: top-to-bottom CAB, cycle ABC, order_axis y, repeat_axes x.
- Counterclockwise 90: top-to-bottom BAC, cycle CBA, order_axis y, repeat_axes x.

Rotate segmentation, object-reference masks, ROI masks and configured lattice
vectors as well as images. Top valid_height becomes a side ROI after a quarter
turn; use the rotated valid_mask instead of reusing height1891. Relative angles
need not change when source and target both rotate together.

## Verification

Observed RED before implementing the axis adapter. New tests include:
- Both quarter-turn directions, all three cyclic phases and both clipped edges:
  labels and identity maps exactly equal the expected rotated synthetic truth.
- Project each exported original-coordinate object matrix onto its actual source
  mask and compare with its predicted identity region.
- Rotated white-defect support completion, repeat_axes disabled, and nested
  diagnostic matrix/vector/center roundtrip checks.
- Rotated LabelMe points, explicit y-layout JSON workflow and side ROI masks.
- Conflicting repeat/layout axes and mixed template axes rejected.
- Actual current SD-1 templates/input, without mocked image registration:
  baseline CAB; clockwise CAB; counterclockwise BAC; both accepted, zero unknown
  ROI pixels, excluded side strips remain invalid.

All 121 tests pass, none skipped. Signature compatibility confirmed for all
12 callers of from_options; editor diagnostics clear on changed code.

Actual rotated-mask consistency against rotated original predictions:

| Rotation | Class 1 IoU | Class 2 IoU |
|---|---:|---:|
| Clockwise | 0.993042 | 0.923978 |
| Counterclockwise | 0.999036 | 0.916155 |

These are not GT accuracy. Full ECC registration, discrete working-resolution
peaks and alternate reference selection are not exactly rotation-equivariant;
the actual-mask tests require >0.90 class consistency and >0.85 per-identity
consistency, while controlled synthetic tests require pixel-exact output.

Saved actual rotated outputs in results/rot90_cw_current and
results/rot90_ccw_current. Inspected results/rotation90_comparison.jpg:
left column = rotated baseline, right column = actual independent registration;
top row = clockwise, bottom row = counterclockwise.

Re-ran current unrotated config and refreshed results/sd1_opencv: accepted CAB,
zero unknown pixels. Final labels/validity, base labels/validity and object IDs
are pixel-exact to the pre-adapter class_schema_legacy_current baseline.