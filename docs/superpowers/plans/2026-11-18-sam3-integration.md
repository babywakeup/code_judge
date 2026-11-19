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