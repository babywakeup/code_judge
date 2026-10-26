# OpenCV matcher backend

## Goal
Add an OpenCV template-matching backend selectable from JSON configuration,
while preserving the existing FFT backend and its default behavior.

## Approaches
1. Replace only coarse translation correlation with cv2.matchTemplate
   (recommended). Reuse angle/scale enumeration, candidate selection, ECC,
   scoring, label transfer and report generation for comparable results.
2. Build a separate complete matcher. This duplicates registration and output
   logic and makes backend comparisons harder; not recommended for this scope.

## Proposed design
- Add MatcherConfig.backend with accepted values "fft" (default) and "opencv".
  JSON selects it using options.backend; the Python API uses the same field.
- Keep the current FFT translation search unchanged.
- Implement the OpenCV translation search in a private internal module.
  Pad the target to support negative shifts and partial overlap, then use
  cv2.matchTemplate with TM_CCORR to compute correlation, valid overlap and
  overlap-dependent feature energies. Normalize only on real valid pixels;
  padding must not count toward coverage or similarity.
- Apply the same minimum-overlap tests, coverage penalty and peak suppression
  as the existing backend. Return the same (score, dx, dy) candidate format.
- Share downstream affine refinement, final scoring, acceptance criteria,
  segmentation transfer and report formats. Result config records the backend.
- Keep config.sd1.labelme.json explicitly on fft and provide a separate OpenCV
  trial config with the same images and annotations.

## Scope
This adds an alternative global registration backend. Periodic annotation
extension and class-1 anchor detection are separate future features.

## Validation
Test positive and negative translations for equal-size images, partial overlap,
invalid regions, and unequal image sizes. Compare coarse-search candidates
against FFT on deterministic fixtures. Verify rejection of unknown backends,
label transfer through the public API and JSON backend routing. Run the full
existing test suite and the real SD-1 trial into a new output directory.

## Tradeoff
OpenCV correlation may be slower for large search canvases. Numerical scores
can differ slightly between implementations; final scoring remains shared.

Status: approved by user ("do it"); implemented.
