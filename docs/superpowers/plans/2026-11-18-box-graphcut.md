# Box Graph Cut Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce defect contours from detector boxes and reference anomaly evidence without GT leakage.
**Architecture:** Separate array graph-cut predictor, prediction CLI, and post-hoc GT evaluation module. Persist prediction with source/config hashes before evaluation.
**Tech Stack:** Existing NumPy, OpenCV and SciPy sparse maximum_flow.
**Spec:** docs/superpowers/specs/2026-11-18-box-graphcut-design.md

## Global Constraints
- No GT file/contour access during prediction; evaluation only after prediction saved.
- No GT-driven choice of seeds, thresholds, masks or hyperparameters on this image.
- Preserve inputs/config and all existing anomaly branches.

## Review Focus
- Source/sink capacities must correspond to the intended foreground probability.
- No seeds and no reliable reference must return explicit abstention.
- XML boxes may touch/exceed edges; no forced clipping to the original detection box.
- Multiple foreground components and color diversity cannot be replaced by a white-only prior.
- Evaluation must preserve frozen prediction and count missing reference coverage as potential FN.

### Task 1: Predictor
- [x] Write failing tests for mincut direction and synthetic seeded refinement.
- [x] Implement box parsing, anomaly/appearance unaries, pairwise graph and prediction exports.
- [x] Verify synthetic cases and GT-path independence.

### Task 2: Evaluation and real run
- [x] Write failing tests for polygon rasterization and pixel metrics.
- [x] Implement independent evaluator, source matching and GT comparison artifacts.
- [x] Freeze actual prediction, evaluate once, inspect overlay, run regressions and review.

Execution: user approved the proposed baseline; implement in current feature branch and retain uncommitted prior work. No publishing or commits requested.

## Execution record (2026-11-18)

- Added array predictor `reference_anomaly/box_graphcut.py`, independent prediction CLI
	`reference_anomaly/box_prediction.py`, post-hoc evaluator `reference_anomaly/box_evaluation.py`.
- Default fixed algorithm: 10%/16px context, salience >=3 reliable FG components (max 16,
	max 32 seeds each), reliable score <=1 outside all boxes for BG, signed Lab residual
	multi-prototype appearance, contrast-sensitive 4-neighbor mincut. No GT inputs to predictor.
- Actual prediction saved to `periodic_template_matcher/results/sd1_opencv/anomaly/box_segmentation/`
	BEFORE first actual GT access. Manifest SHA256:
	`0fa28d47a2d72212c27aef812735435d34e88b878e80d179bd2b7cd0358e6c86`.
- First evaluation attempt rejected embedded LabelMe imageData before computing metrics.
	Extended evaluator only (not prediction): exact external-image identity is required when available;
	embedded copy discrepancy is recorded. Actual external image equals frozen input exactly.
	One completed actual evaluation; no mask/seed/threshold changes after seeing GT.
- Prediction 36,540 px; TP 21,783, FP 14,757, FN 23,276, TN 5,006,168.
	Precision .5961412, Recall .4834328, IoU .3641668, Dice .5339036.
	GT reference coverage 100%, 0 uncovered GT px. Explicit warnings: touches_box,
	touches_roi, touches_unknown_reference, seed_components_truncated. Baseline is not
	reliable precise segmentation; retain failures instead of GT-tuning this image.
- Evaluation saved separately to sibling `box_segmentation_evaluation/`; frozen artifact
	hashes verified before GT read, source/GT/prediction overwrites prohibited.
- Full Python regression initially passed: 226 tests / 82.058 s; review found no concrete
	correctness issue. Final regression after exhaustive-energy/mixed-color contract tests:
	**228 tests passed / 153.021 s**, including 28 box-prediction/evaluation tests.
	Final verification confirmed every frozen artifact hash, manifest hash and predictor/exporter
	code hash remained unchanged since prediction. Editor/Pylance checks reported no errors.
