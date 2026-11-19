# SAM3 local image integration — execution record

## Scope and separation

- Keep matcher normal reference/anomaly evidence and graph-cut baseline unchanged.
- Add two controls: detector box only; detector box plus reference-anomaly point prompts.
- Original RGB crops go to SAM3; anomaly scores guide prompts only. Do not intersect
  model masks with graph-cut/threshold/reference-validity masks or clip to detector boxes.
- No GT access during prompt preparation or prediction. Freeze predictions before evaluation.
  This image is already a development example, not an unseen final test.

## Completed

- [x] Verify local weights and runtime facts.
- [x] Array-only deterministic prompt planner and tests.
- [x] Optional official SAM3 adapter with local checkpoint and no HF auto-download.
- [x] New-only prompt-preview and prediction CLI; preserve candidates and selection provenance.
- [x] Extend frozen-artifact evaluator to accept sam3_prediction, not prompt preparations.
- [x] Generate actual box and anomaly-point previews, no GT read or model inference.
- [x] Run regression tests and document runtime limitations.

## Verified API

Official model builder supports checkpoint_path, load_from_HF=False,
enable_inst_interactivity=True, eval_mode=True, device and compile=False.
Sam3Processor(model, device=...).set_image(PIL RGB) -> state;
model.predict_inst(state, point_coords=Nx2 pixels, point_labels, box=XYXY,
normalize_coords=True, multimask_output=True, return_logits=False) -> masks, quality, logits.

The upstream SAM2Transforms.transform_coords explicitly DIVIDES pixel coordinates
by original W/H when normalize=True, then scales to model resolution. A review
suggestion to reverse that flag was checked against upstream source and rejected.
The adapter contract test checks this flag and pixel inputs.

## Runtime evidence (2026-11-18)

- weights/sam3.pt: 3,450,062,241 bytes; model.safetensors: 3,439,938,512 bytes.
  Official backend selects .pt; no checkpoint deserialization was attempted.
- Selected interpreter .venv: Python 3.12.13; Torch 2.14.1+cpu; CUDA runtime None,
  torch.cuda.is_available() False; official sam3 package absent.
- Windows video controller: Intel UHD Graphics 770 (also OrayIddDriver), no detected NVIDIA GPU.
  No driver/CUDA/Torch replacement attempted; no unverified CPU inference claim.
- Prompt previews saved under existing result anomaly/sam3_box_prepared and
  anomaly/sam3_anomaly_points_prepared. Fixed 10%/16px crop; <=16 high-response
  components; <=3 positive points/component and <=8 safe negative points.
  Actual crop xyxy=[1040,289,1658,850], 16 guided tasks, seed_components_truncated warning.
- Preparations have kind=sam3_prompt_preparation, model_executed=False, no predicted masks.
  An adapter comment was clarified after preview generation; snapshots retain code hashes
  from preparation time. No prompts or constants were changed after viewing the previews.

## Historical validation (official adapter, November 18)

- 28 new synthetic tests; 56 related tests passed. Full Python regression:
  **256 tests passed / 89.740 s**. Editor/Pylance checks clean.
- Synthetic backend is identified explicitly as a test double; no actual SAM3 accuracy
  or speed is reported. Actual GT files were not accessed in this integration task.
- At this stage the adapter only supported the official repository. Its environment
  limitations do NOT apply to Transformers. See the corrected path below.

## Transformers CPU correction (November 19)

- [x] Inspect installed Transformers 5.21.0 source and the local HF bundle.
- [x] Add an optional offline Transformers adapter, without importing Torch at module import time.
- [x] Default backend auto prefers local model.safetensors; default device auto selects
  CUDA only when available, otherwise CPU. Explicit CPU uses FP32 and inference_mode.
- [x] Box/text: Sam3Model + Sam3Processor, followed by post_process_instance_segmentation.
  These masks are DISTINCT INSTANCES: union all threshold-filtered nonempty instances.
- [x] Box/points: Sam3TrackerModel + Sam3TrackerProcessor, cached image embeddings,
  post_process_masks, select the highest-quality nonempty point-consistent alternative.
  Never silently discard point prompts or send unsupported points to Sam3Model.
- [x] Local files only; hash HF configs/tokenizer/checkpoint and adapter code. Reject
  missing/mismatched model parameters instead of using random initializations.
- [x] Add dependency-free mocked HF contracts, multi-instance union and abstention tests.
- [x] Attempt real local CPU execution through the dedicated VS Code task.

### Actual execution evidence

- The bundle config is sam3_video with detector_config and tracker_config; installed
  image/tracker constructors support extracting their respective subconfigurations.
  This source-level check is NOT a successful weight-load verification.
- Actual Transformers CPU loading failed with Windows pagefile/commit-limit error 1455.
  At observation time available commit headroom was approximately 2–3 GB.
- No real prediction manifest exists, no real SAM3 accuracy/latency is reported, and no
  actual GT was read during this work. Tracker loading and inference are also unverified.
- CUDA and the official sam3 package are NOT required for this Transformers CPU path.
  Do not repeat the earlier incorrect claim or replace the CPU Torch environment.
- Remaining runtime prerequisite: enough system memory/commit capacity. No unrelated
  processes were terminated and no pagefile, driver or system settings were changed.

## Continuation on feat/supposed-to-be-good (November 19)

- Confirmed the correct branch and existing Transformers integration. Fixed only the
  synthetic HF-bundle test fixture: its config.json previously overwrote the business
  target/box config. Use a separate hf_bundle directory and assert config preservation.
- Related 61 tests passed in 0.574 s. Full Python regression: **271 passed / 86.020 s**.
  Editor diagnostics were clean for both adapters/prediction and their tests.
- NumPy testing's platform.machine() hit an unresponsive Windows WMI query, verified
  with a timed Python stack dump. Validation runners used process-local
  platform._wmi_query -> OSError to invoke Python's ordinary Windows fallback.
  No production workaround or system-service change was made. The normal VS Code
  inference task stalled without output and its own two Python processes were stopped.
- Available commit headroom before retry was 7.73 GiB (not the earlier 2–3 GB).
  Real CLI runs via the same process-local WMI fallback then SUCCESSFULLY loaded and
  executed both local models on CPU FP32. Model computations were not mocked.
  Both manifests have model_executed=true, missing_keys=[], mismatched_keys=[].
  Tracker emitted the sam3_video -> sam3_tracker architecture-subset warning;
  complete loading and actual prediction nevertheless succeeded.
- Box image inference: 1 instance, 22,742 predicted pixels, including 271 without
  reference support. Manifest SHA256:
  7838086fc833ff2617e1971ba139ab1552c188283b332bfbaafc95fb6d52887f.
- Guided tracker inference: 16 fixed component tasks, all status ok; seed component
  truncation retained and reported. Union: 182,715 pixels, 679 without reference.
  Manifest SHA256:
  4c629106abc677658578be5bfd91b47de002ecbb7d74c887d255959b3c067258.

### Independent evaluation after both predictions were frozen

Both manifest hashes were recorded BEFORE GT access. The existing independent
evaluator verified all artifact hashes before reading GT and created new separate
sam3_box_transformers_evaluation / sam3_anomaly_points_transformers_evaluation
directories. Manifest hashes were checked unchanged afterwards. External decoded
original image equality passed; embedded-copy-differs warning is retained. Full-image
GT coverage is 100%. No prompt, threshold, candidate or output mask was changed.

| Method | TP | FP | FN | Precision | Recall | IoU | Dice |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical GraphCut | 21,783 | 14,757 | 23,276 | 59.61% | 48.34% | 36.42% | 53.39% |
| SAM3 image box | 4,022 | 18,720 | 41,037 | 17.69% | 8.93% | 6.31% | 11.86% |
| SAM3 tracker box + anomaly points | 44,675 | 138,040 | 384 | 24.45% | 99.15% | 24.40% | 39.23% |

The current SAM3 integration runs correctly but does NOT improve the baseline on
this known development image. Guided prompts achieve high recall at substantial
over-segmentation. Image and tracker modes differ in model interface as well as
prompts, so this is not a controlled ablation proving normal reference is necessary
or redundant. Do not tune against these GT results; any new prompt policy requires
fixed rules and separate validation data. No claim of unseen-test performance or
measured inference latency is made.

## User-requested larger image context (November 19)

- Add --context-fraction (API context_fraction) with default 0.1 unchanged. This
  scales padding PER SIDE by original detector width/height, with the existing
  16px minimum and image-boundary clipping. Original detector prompts are only
  translated to crop coordinates, never enlarged. Record the actual fraction in
  the frozen manifest. Finite nonnegative values only; very large values saturate
  at image boundaries without multiplication overflow.
- Tests cover expanded crops in both modes, original box/point preservation,
  edge clipping, invalid fractions, RGB input and full-image output mapping,
  no output clipping back to prompt/validity, and CLI parameter forwarding.
  Related **66 passed / 0.591 s**; full **276 passed / 83.399 s**. Editor diagnostics
  clean. Existing production calls compatible; prompt-test signature warnings
  concern deliberate malformed validation inputs and heterogeneous test dictionaries.
- Execute ONE real local Sam3Model CPU FP32 box-only prediction with fixed
  context_fraction=0.5 (not a GT-driven sweep). No text/points, same 0.5 confidence
  and mask thresholds. Use the existing process-local Windows WMI fallback only.
- Original detection [1092,336,1606,803] unchanged. ROI becomes
  [835,102,1863,1037], 1028x935 instead of 618x561. Local prompt is
  [257,234,771,701]. Complete weights load: missing_keys=[], mismatched_keys=[].
- New-only anomaly/sam3_box_transformers_context50/prediction.json; SHA256
  39cf9d2f45221ad794762fd381784d58b5d0b33631bb38f96e6c690280b9bb6e.
  model_executed=true, status=ok. Union all 4 threshold-filtered instances:
  119,937 pixels, 132 without reference support; touches_roi warning retained.
  Overlay visibly includes repeated structures outside the detection as well.
  Do not claim improved accuracy from increased area or visual inspection.
- Both previous prediction manifest hashes verified unchanged. No new GT read or
  evaluation in this user-requested run. This remains a known development image,
  not a new held-out experiment; no candidate selection or threshold tuning.