"""Coverage-aware metrics and their evaluation step; independent of registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config import MatcherConfig
from ...utils import ValidRegion, normalize_masks, validate_image
from ...utils.types import BoolMask, Diagnostics, Image, LabelInput, LabelMap, Mask
from ..models import EvaluationResult, MatchResult
from .preparation import InputPreparationStep


def _metrics(prediction: LabelMap, truth: LabelMap, known: BoolMask,
             roi: BoolMask, include_background: bool) -> Diagnostics:
    # Unknown predictions are abstentions, not class 0: FN for the true class,
    # no FP for any predicted class, and incorrect for pixel accuracy.
    ids = np.union1d(np.unique(truth[roi]), np.unique(prediction[roi & known]))
    if not include_background:
        ids = ids[ids != 0]
    per_class = []
    for value in ids:
        actual = (truth == value) & roi
        predicted = (prediction == value) & known & roi
        tp = int(np.count_nonzero(actual & predicted))
        fp = int(np.count_nonzero(predicted & ~actual))
        fn = int(np.count_nonzero(actual & ~predicted))
        union = tp + fp + fn
        per_class.append(dict(class_id=int(value), tp=tp, fp=fp, fn=fn,
                              iou=tp / union if union else None,
                              dice=2 * tp / (2 * tp + fp + fn) if union else None))
    ious = [c['iou'] for c in per_class if c['iou'] is not None]
    dices = [c['dice'] for c in per_class if c['dice'] is not None]
    count = int(np.count_nonzero(roi))
    correct = int(np.count_nonzero((prediction == truth) & known & roi))
    return dict(mean_iou=float(np.mean(ious)) if ious else None,
                mean_dice=float(np.mean(dices)) if dices else None,
                pixel_accuracy=correct / count if count else None,
                evaluated_pixels=count, per_class=per_class)


def evaluate_segmentation(prediction: LabelMap, truth: LabelMap,
                          prediction_valid: Mask, truth_valid: Mask, *,
                          include_background: bool = False) -> Diagnostics:
    """Score full GT ROI; also expose overlap-only metrics and abstention rate.

    All inputs must be same-sized 2D arrays. Classes absent from both prediction
    and truth are omitted; no foreground and no background scoring => score=None.
    """
    if any(x.shape != truth.shape for x in (prediction, prediction_valid, truth_valid)):
        raise ValueError('Prediction, truth and validity masks must have identical shapes')
    known, roi = prediction_valid > 0, truth_valid > 0
    overall = _metrics(prediction, truth, known, roi, include_background)
    covered = int(np.count_nonzero(known & roi))
    count = overall['evaluated_pixels']
    return dict(status='evaluated' if count else 'no_valid_ground_truth',
                score=100 * overall['mean_iou'] if overall['mean_iou'] is not None else None,
                score_definition='100 * mean_iou over ground-truth ROI; unknown predictions are abstentions',
                include_background=include_background, **overall,
                prediction_coverage=covered / count if count else None,
                covered_pixels=covered, uncovered_pixels=count - covered,
                overlap_only=_metrics(prediction, truth, known, roi & known, include_background))


@dataclass
class PreparedGroundTruth:
    labels: LabelMap
    valid: BoolMask


class SegmentationEvaluationStep:
    """Validate GT before matching, then score the finished prediction only."""

    def __init__(self, config: MatcherConfig) -> None:
        self.config = config

    def prepare(self, target: Image, labels: LabelInput, input_valid: Mask | None,
                target_valid: Mask | None, target_valid_height: int | None) -> PreparedGroundTruth:
        if not isinstance(target, np.ndarray):
            raise ValueError('input must be a numpy image array')
        validate_image(target)
        truth = normalize_masks(labels, target.shape[:2], 'input_masks')
        InputPreparationStep.validate_masks([input_valid], [target], 'input_valid')
        roi = ValidRegion.from_config(self.config, target_valid_height).build_mask(target, target_valid) > 0
        if input_valid is not None:
            roi &= input_valid > 0
        return PreparedGroundTruth(truth, roi)

    def run(self, matched: MatchResult, truth: PreparedGroundTruth) -> EvaluationResult:
        if matched.mapped_labels is None:
            metrics = dict(status='match_rejected' if not matched.accepted else 'no_template_annotations',
                           score=None, mean_iou=None, mean_dice=None, pixel_accuracy=None,
                           prediction_coverage=0.0 if truth.valid.any() else None,
                           evaluated_pixels=0, per_class=[], overlap_only=None)
        else:
            metrics = evaluate_segmentation(matched.mapped_labels, truth.labels, matched.label_valid,
                                            truth.valid, include_background=self.config.include_background)
        return EvaluationResult(matched, metrics, truth.labels, truth.valid)
