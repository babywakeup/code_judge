# Explicit valid heights and modular region handling

Goal: exclude bottom device text in templates and input before registration;
centralize mask semantics in ValidRegion instead of repeating height logic.

Design: ValidRegion(valid_height=None, bottom_ignore_fraction=0) builds an original-
size uint8 mask, intersects a user mask and validates sufficient usable area.
Explicit height overrides fraction. Original pixels [0,H) are valid rows. A crop
helper is available to callers but registration keeps original coordinates.
MatcherConfig.valid_height is a common default; per-template valid_height and
target_valid_height override it in JSON. Array match/evaluate accept
template_valid_heights and target_valid_height. Metadata records resolved rows.
Masks are resolved once at public matcher entry; internal registration uses a
config without implicit height/fraction to avoid applying exclusion twice.
Annotation validity remains independent and is intersected with image ROI.

Plan:
- [x] Write failing tests: class behavior, precedence, invalid heights, mask holes,
  independent heights, footer invariance, evaluate ROI, config/default/dedup routing.
- [x] Implement ValidRegion class; preserve build_valid_mask compatibility wrapper.
- [x] Prepare masks before matching; reuse resolved masks through label transfer,
  completion and metrics. Update workflow and metadata/report.
- [x] Migrate SD-1 configs to explicit 1891-pixel valid heights.
- [x] Run full tests and both real-image configs, verify original-size exports.
- [x] Independent review and fix findings.

Verification: six initial test methods failed before implementation, then passed.
Extended completion test confirms independent source/target heights and mask holes.
Independent review reproduced excluded-footer contamination during INTER_AREA
downsampling; new regression failed in both backends. ValidRegion now performs
valid-pixel weighted resize before feature extraction, reused by registration and
completion. Canny uses repaired image as well. Regression GREEN; full suite 49/49.
Both real SD-1 runs accepted (80.61), complete valid ROI at H=1891. Exported PNGs
retain 2056x2464 shape, all valid rows covered and excluded footer zero.
