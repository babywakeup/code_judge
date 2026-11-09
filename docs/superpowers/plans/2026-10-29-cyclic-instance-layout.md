# Cyclic instance layouts

User specification: class 2 contains N different instances, each appearing once
per horizontal group. Templates supply their left-to-right identities. Inputs
may show cyclic rotations and clipped edge instances. Class 1 must retain its
existing registration and periodic completion behavior.

Design: retain the existing image candidate generator. Learn directed adjacent
identity spacings from original template masks, transformed into target pixels.
Solve N rotations by dynamic programming with appearance and spacing costs.
Compare different rotations as before; do not introduce same-rotation reference
ambiguity into the existing ABC acceptance policy. N=1 compares locations.
Never assume equal spacing, a one-pixel period, or infer invisible masks.

Preserve the base labels/validity as separate arrays before merging class 2.
Accepted class-2 foreground supplements base background; overlaps with existing
foreground are reported and do not overwrite class 1. Class-2 uncertainty remains
separate from base knowledge. Save base artifacts and final combined labels.

Tasks and checks:
1. RED tests for five cyclic rotations, clipped boundaries, unequal spacings,
   ambiguous appearance resolved by spacing, and invalid template identities.
2. Implement N validation and a bounded DP with learned adjacent spacing.
3. Test and implement preservation of actual periodic-completion output through
   accepted, ambiguous and partial-reference object branches; save/load artifacts.
4. Explicitly activate known MA ABC / UA CAB configuration, preserving target and
   all registration settings. Run full tests and several actual inputs.
5. Produce and inspect base, object and combined overlays; record failures as
   failures rather than loosening thresholds or claiming visual accuracy from
   valid-ROI coverage alone.

Validation completed:
- 85 tests pass, including all five rotations and both target-edge fragments,
  unequal-gap phase discrimination, exhaustive DP optimality, N=1 ambiguity,
  five-instance LabelMe/workflow/export/viewer routing, and actual periodic-fill
  preservation under accepted/ambiguous/partial object outcomes.
- Original current1343428 accepts ABC, margin 0.0241196 (unchanged gap threshold
  0.02); 1379453 accepts ABC, margin 0.0437377; 1383316_defect accepts BCA,
  margin 0.0283321. These are matcher decisions, not GT accuracy measurements.
- Each actual input was independently rerun with class2 removed and the object
  branch disabled. Base label arrays, base validity, and final class1 foreground
  are pixel-exact matches to that independent baseline.
- Inspected three-panel base/object/combined overlays for all three inputs.
  Current output is results/sd1_opencv, including qa_layers.jpg. Individual
  regression artifacts are in results/cyclic_*.
- Read-only independent review found no concrete new correctness/regression
  issue. git diff --check passes (only existing newline-conversion warnings).

Limitations retained: a completely invisible instance is not synthesized;
missing template neighbor spacings do not become assumed wrap distances;
unobserved object support stays unknown. Final class2 foreground conflicts
with established base foreground are clipped and counted, rather than silently
overwriting class1. The exported base artifacts preserve the unclipped base
pipeline independently of final combined background knowledge.

Follow-up: current input's left gap was missing vertical support of confirmed A,
not missing class1 completion. A's only initial candidate translates source y by
-766 and covers 59% vertically; no alternate A peak passes the candidate gate.
The verified base lattice yields target vertical period 871px. Added a separate
object_completion module: same identity, near-vertical target lattice only,
per-region appearance/edge verification, no horizontal copies or identity change.
Current gap restores final ROI coverage from 97.7949% to 100%, adding 116972
object-support pixels while base labels/validity and class1 foreground are exact
matches to the prior run. New tests reject unmatched missing patches and
horizontal-period copies. Independent review found that contextual scoring alone
could approve noisy 1/5-row edge strips. Added original-resolution correlation
inside each actual fill region (>=64 pixels, nonflat source and target), alongside
the contextual checks; reproduced and fixed both cases. 89 tests pass and
follow-up review confirms the issue resolved. Final current result has zero
unknown ROI pixels, with 116972 added support pixels and 29136 class2 pixels;
class1 and the base outputs remain pixel-exact. Final artifacts and refreshed
qa_layers.jpg / left_completed.jpg are in results/sd1_opencv.

## White-defect segmentation transfer follow-up (2026-10-29)

Clarified user requirement: transfer the healthy template segmentation even
under a white defect; a defect is not evidence that the underlying class is
unknown. This supersedes the earlier test that deliberately left white patches
unknown. Identity/geometry still require visible image evidence; no horizontal
instance copying or synthesis beyond observed source annotation is introduced.

Cause: C can be identified from a bottom fragment, but its vertical extensions
were rejected because original-resolution region NCC compared healthy texture
against white occlusion. Local tile retries recovered visible areas but left
80,239 unknown ROI pixels in the fresh pre-fix defect run (98.2779% coverage).

Fix in object_completion: separate target appearance evidence from output ROI.
Exclude connected near-white patches (gray >=230, area >=64 original pixels)
from feature and direct-pixel validation, but not from label transfer. Each new
region still requires >=64 nonflat visible pixels covering at least half the
region and the unchanged NCC/structure/edge gates. The same verified transform
then transfers the entire source label region, including occluded foreground
and background. Features are scored on the intersection of source support and
the target evidence-safe mask, not on inpainted excluded pixels. Evidence and
occlusion counts are exported per region. Explicit target ROI and source
annotation support remain authoritative.

Verification:
- Observed RED for the new white-occlusion full-label transfer test, then GREEN.
- 93 tests pass; random mismatch, noisy thin strips, horizontal-copy rejection,
  fully unsupported white regions, near-white values 230/245/255 and explicit
  target ROI exclusion are covered.
- Entry configuration now targets Tlink-Image-1792571383316_defect.jpg; run.sh
  itself is unchanged because it already invokes this config and the viewer.
- Fresh defect result: BCA accepted, zero unknown ROI pixels (100% coverage).
  14,139 C-identity foreground pixels transfer onto near-white image pixels.
  Base labels, base validity and final class-1 foreground are pixel-exact to
  the pre-fix baseline. Coverage is not a segmentation-GT accuracy metric.
- Re-ran 1343428 and 1379453 with and without the new evidence exclusion:
  both accept ABC, zero unknown ROI pixels, identical final labels/validity
  and base labels.
- Saved before/after results under results/defect_transfer_before and
  results/sd1_opencv; inspected defect_transfer_comparison.jpg (orange = unknown,
  green = class 1, purple = class 2). git diff --check passes.
