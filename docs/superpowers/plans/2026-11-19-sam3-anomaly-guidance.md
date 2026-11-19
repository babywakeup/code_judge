# Fixed anomaly-assisted SAM3 tracker experiment

## Predeclared scope (before model execution)

- Preserve existing box/image and per-component anomaly_points modes and outputs.
- Add tracker_box (no anomaly inputs) and anomaly_guided (new evidence policy).
  Both real controls use the same local Sam3TrackerModel, original RGB crop,
  original detector prompt and context_fraction=0.5. No text. CPU FP32.
- Do NOT read GT, run GT evaluation, or change rules after inspecting predictions.
  The current image is a known development example, not a held-out test.

## Fixed guidance policy v1

- Only confidence=255 reference pixels are trusted. Scores >=3 supply foreground;
  scores <=1 supply normal evidence. Intermediate scores and confidence 0/128
  are neutral, not background. No white-color prior or GT-derived thresholds.
- Inside the original detection, retain up to 16 high-score 8-connected components,
  ordered by descending peak then component ID. Within each, the deepest half of
  the distance-transform range (at least depth 1px) defines the anomaly core.
  Spread up to 3 positive points per selected core. Report component truncation.
- ONE joint prompt per box contains all selected component positives, rather
  than independently unioning one mask per component.
- Negative candidates must be reliable low-score pixels at least 3px away from
  ANY reliable high-score region in the crop. Restrict negative prompts to within
  max(8px, ceil(0.1 * min(clipped detector width, height))) of high-score regions.
  Spread up to 8 negatives inside the box and up to 8 outside ALL detections.
  Other anomaly components are never negative. If no negatives are available,
  abstain; if one pool is unavailable, report it and use the other pool.
- All candidate masks must be nonempty and respect every point. Require >=50%
  coverage of selected core pixels, and <=50% normal fraction among mask pixels
  in the selected-core OR reliable-normal evidence domain. The 3px safety zone
  also applies to the reliable-normal scoring map.
- Rank accepted alternatives by core_coverage - normal_fraction, then model
  quality, then stable index. Reject all -> explicit abstention, not fallback to
  a weaker policy. These evidence scores are NOT calibrated GT IoU.
- Never modify SAM masks by heatmap intersection, validity clipping, detector
  clipping, smoothing or hole filling. Save all raw masks, evidence maps,
  candidate metrics, rejection reasons, fixed constants and hashes.

## Verification completed

- [x] Synthetic prompt, unknown-reference neutrality, candidate rejection/ranking,
  abstention, original-box preservation, freeze and GT-isolation tests.
- [x] Full Python regression and editor diagnostics.
- [x] New-only real tracker_box and anomaly_guided predictions at the same context.
  Check all artifact hashes and preserve historical frozen manifests.
- [x] Record execution and prediction facts honestly; no GT accuracy claim.

## Verified execution (November 19, after stash restoration)

- Re-read all externally changed prompt/prediction/test files before continuing.
  New modes and fixed policy remained intact; preserve unrelated .gitignore edits.
- Related 82 tests passed / 0.758 s. Full Python regression after restoration:
  **292 passed / 89.403 s**. Relevant editor/Pylance diagnostics clean.
- The first real control was correctly rejected BEFORE model construction:
  original target does not match the saved matcher input. Current input and
  matcher manifest differ from the historical frozen hashes, while the original
  target, detector XML, anomaly manifest, scores, valid and confidence still match.
  No identity-check relaxation, source overwrite or alignment rerun was performed.
- Established new results/sd1_sam3_guidance_replay with byte-identical frozen input
  and hash-verified score/valid/confidence copies. The small manifests explicitly
  identify a verified historical-evidence replay, NOT a new matcher result or a
  recovered original matcher manifest. The historical acceptance basis is the
  admitted frozen prediction; record its full manifest hash and original matcher
  hash, not the current changed matcher as though it generated these scores.
  Original target decoded pixels equal replay input. No GT accessed.
- Both real runs use this SAME replay manifest, input, local checkpoint and
  Sam3TrackerModel / Sam3TrackerProcessor, CPU FP32; same ROI
  [835,102,1863,1037], original detection [1092,336,1606,803], local prompt
  [257,234,771,701], no text. Both load completely without missing/mismatched keys.
  The known architecture-subset warning is retained. Only process-local WMI
  fallback is used; no model computations are mocked and no system service changed.

### Frozen outputs

All output directories below are under results/sd1_opencv/anomaly, created new:

- sam3_tracker_box_transformers_context50/prediction.json:
  **2a6e084230c568cc06ea06865ae1e5d0bd9fa187a8ca146ec4de2970c80531ad**.
  model_executed=true, status=ok. Alternatives have 50,767 / 181,953 / 104,838
  pixels; model-quality selection chooses index 2 (0.584161), 246 pixels without
  reference. No scores or confidence are read for this control.
- sam3_anomaly_guided_transformers_context50/prediction.json:
  **101f40fd5423f8464657b6b42cff6a1d7ec5fbe64d7d4602697df6f3563cb298**.
  model_executed=true, status=abstained. One joint prompt has 48 positives from
  16 retained components, 8 inside negatives and 8 outside negatives. Component
  truncation warning retained. No accepted candidate, final empty mask.

| Raw guided candidate | Pixels | Missed positives | Covered negatives | Core coverage | Normal fraction | Result |
|---|---:|---:|---:|---:|---:|---|
| 0 | 5,501 | 42 | 0 | 15.59% | 96.45% | Point conflict, insufficient core, excess normal |
| 1 | 73,630 | 11 | 1 | 70.40% | 98.86% | Point conflict, excess normal |
| 2 | 71,664 | 19 | 1 | 61.92% | 99.01% | Point conflict, excess normal |

These are reference-evidence fractions, NOT GT metrics. All candidates already
violate points, so merely relaxing evidence gates would not produce a valid
candidate under this fixed policy. Abstention means **no trustworthy segmentation**,
NOT a no-defect classification. No success/improvement/accuracy claim is warranted.
Do not tune or re-prompt from this observed output in the current experiment.

### Integrity and diagnostics

- Verified all frozen artifact hashes in both new outputs; checked equal input,
  checkpoint, replay manifests, reference valid, full backend metadata, ROI and
  original detector prompt. Historical three prediction manifest hashes unchanged.
- New-only sam3_anomaly_guided_context50_diagnostics/rejected_candidates.png shows
  the 3 REJECTED raw alternatives and point-conflict counts. It is not a prediction;
  its diagnostics.json records the source manifest hash and image hash
  df5cc9edfaa469a06cce0e47dd4dac79277a0a345966ab8f0d365a54b7373652.
  No frozen mask/candidate was modified to produce the diagnostic.
- No actual GT read, no new GT evaluation, no parameter sweep, no measured
  latency claim. Current image remains a known development sample.