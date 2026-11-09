# Partial instance cycle implementation plan

> Execute inline using superpowers:executing-plans; verify with TDD and a focused final review.

**Goal:** Match three consecutive visible identities from ABCD using all template references, including clipped edge instances.

**Architecture:** Separate complete identity cycle from target visible count. Accept contiguous partial template layouts. Pool per-reference appearance candidates from all templates; rejected whole-image registration may still contribute locally verified candidates using configured angle/scale hypotheses. Solve fixed-length cyclic windows and fuse only observed target support. Preserve legacy full-cycle behavior when visible_count is omitted.

**Scope:** class 1 retains existing base registration/periodic behavior. No horizontal instance replication or invented fourth instance. Keep class 2 results usable when only its local evidence is accepted, without transferring unapproved base labels or asserting background knowledge.

- [x] Add partial-cycle tests: all four windows, clipped edges, complementary templates, rejected global geometry fallback, ambiguity, invalid/noncontiguous layouts, configuration, and partial-only result export.
- [x] Implement visible_count schema and partial reference validation.
- [x] Implement window-length layout scoring and all-reference candidate generation/provenance.
- [x] Integrate object-only evidence when base registration is unavailable; preserve legacy pipeline behavior.
- [x] Update current config (cycle ABCD, visible_count 3) with user-confirmed template orders ABC and DAB.
- [x] Run focused and full tests, real data inference and inspect diagnostics; update README/architecture and review.

**Data resolution:** user confirmed the two configured template orders as ABC and DAB. Other files in templates2 remain unconfigured because their identity assignments are unknown.

**Verification:** 51 focused tests passed. Full suite ran 157 tests with two previously observed failures in the real-data clockwise/counterclockwise rotation subtests (the full-ROI-valid assertion at test_rotation_equivariance.py:209). These assertions were not weakened. Current real-input inference accepted ABC, with A/C selected from template 0 and B from template 1; both sources contributed candidates (19 and 24). This is inference evidence, not ground-truth accuracy validation. Output: periodic_template_matcher/results/sd1_partial_cycle.

**Review:** attempted independent review could not run because the reviewer hit a usage limit; performed local review of schema, coordinate adaptation, layout, pipeline and export behavior instead.

**Identity correction:** the user subsequently confirmed the actual target is DAB, with D clipped at the left boundary. The earlier ABC acceptance was incorrect: vertical aliases at two interior positions exhausted the eight correlation peaks before the real edge D could enter layout comparison. Partial-cycle peak extraction now caps repeats per horizontal neighborhood and expands its bounded search budget to retain distinct positions. Existing appearance and visibility thresholds remain unchanged. The corrected real run accepts DAB (score 0.7760, next ABC 0.7352, margin 0.0408), with no unknown pixels inside the configured ROI. Added a synthetic candidate-starvation regression and a user-confirmed real-image assertion. Corrected artifacts: periodic_template_matcher/results/sd1_identity_fixed.

**Final verification after identity correction:** unittest discovery passed all 161 tests in 89.156 seconds, including real-image clockwise/counterclockwise rotation checks and explicit absence checks for the invisible fourth identity.
