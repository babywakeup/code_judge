# Class-bound configuration and template layouts

## Requirement

A bare template object_order list does not tell readers which semantic class it
belongs to, which spatial direction defines the order, or which directions admit
same-identity translation symmetry. Make these relationships explicit without
changing matching thresholds or current input/output behavior.

## New JSON structure

- options.classes[class_id].repeat_axes declares that class's permitted repeat
  directions. [] disables copies, not direct segmentation transfer.
- options.classes[class_id].instance_matching declares an identity cycle and
  optional min_score, min_gap, min_visible, candidates and spacing_weight.
- templates[].instance_layouts[class_id] declares order_axis and actual order.
  For UA: class 2, order_axis x, order CAB. Thus the template order and the class's
  repeat_axes y unambiguously mean left-to-right CAB, horizontal nonrepeatability
  and vertical same-identity repetition.

Normalize options.classes to existing MatcherConfig runtime fields. Do not fork
the inference algorithm or change the public array API. Retain legacy JSON
objects/class_periodicity/object_order, but reject mixing conflicting old/new
rule fields and both layout representations within the same template.

The current algorithm only supports one independently matched class and x-axis
identity ordering. Explicitly reject multiple instance_matching classes and
order_axis y, rather than pretending the existing horizontal solver handles them.
Class IDs and the enabled identity class must agree; unused explicit layouts are
errors even when instance matching was accidentally omitted. Repeated direction
rules remain separate from this layout limitation.

## Persistence and duplicate safety

Include instance_layouts in duplicate template annotation signatures. Export
class_rules and normalized template_instance_layouts (including legacy layouts)
to preserve readable provenance. ordered_objects now explicitly reports class_id
and order_axis alongside visible_order and identity results.

## Verification

- Captured actual current-input baseline before changing parser/config under
  results/class_schema_legacy_current: accepted CAB, zero unknown ROI pixels.
- Observed RED for the new configuration tests, then implemented normalization,
  annotation/layout validation, workflow checks and diagnostics.
- Eight new schema tests cover runtime equivalence, non-instance class rules,
  invalid/ambiguous config, explicit class/axis association independent of draw
  order, unsupported layouts, duplicate-image layout conflicts, end-to-end
  workflow/export equivalence, and layouts without enabled instance matching.
- All 113 tests pass; changed code/config editor diagnostics are clear.
- Re-generated results/sd1_opencv for unchanged 1352308_defect input: accepted
  CAB, zero unknown pixels. Compared final labels, label validity, base labels,
  base validity, identities, object validity and periodic label provenance to
  the pre-migration baseline: all seven artifacts pixel-exact.
- README now recommends the new schema, explains the three distinct concepts,
  and documents current capabilities and compatibility behavior.