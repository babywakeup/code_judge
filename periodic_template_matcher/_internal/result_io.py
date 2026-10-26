"""Stage 1: persist lossless prediction data for the independent viewer app."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from ..utils import validate_image, write_image
from ..utils.types import Diagnostics, PathInput
from .results import EvaluationResult, MatchResult


# Only these owned files are replaced on reruns, including obsolete reports.
ARTIFACTS = (
    'result.json', 'input.png', 'mapped_labels.png', 'label_valid.png',
    'label_periodic_valid.png', 'geometric_valid.png', 'matrix_template_to_target.npy',
    'ground_truth.png', 'ground_truth_valid.png',
    'report.html', 'overview.jpg', 'best_alignment.jpg', 'label_overlay.jpg',
    'demo_source_labels.png', 'demo_mapped_labels.png', 'demo_label_valid.png',
    'demo_label_overlay.jpg', 'periodic_overlay.html', 'completed_alignment.jpg',
    'completed_geometric_valid.png', 'periodic_valid.png', 'completion_regions.jpg',
)


def save_result(result: MatchResult | EvaluationResult, output_dir: PathInput) -> Path:
    """Save original-sized arrays and return result.json; never launch a UI.

    Class IDs are stored as uint16 PNG. Validity is separate, so label 0 stays
    distinguishable from unknown pixels. Periodic additions are already included
    in mapped_labels after match(). Rejected results have no prediction files.
    """
    evaluation = result if isinstance(result, EvaluationResult) else None
    match = evaluation.match if evaluation is not None else result
    if not isinstance(match, MatchResult):
        raise TypeError('result must be MatchResult or EvaluationResult')
    validate_image(match.input_image)
    shape = match.input_image.shape[:2]
    has_prediction = match.accepted and match.mapped_labels is not None and match.label_valid is not None
    if match.geometric_valid.shape != shape:
        raise ValueError('geometric_valid shape must match input image')
    if has_prediction:
        if match.mapped_labels.shape != shape or match.label_valid.shape != shape:
            raise ValueError('prediction and validity shapes must match input image')
        if match.mapped_labels.dtype != np.uint16:
            raise ValueError('mapped_labels must be a uint16 class-ID map')
        if match.label_periodic_valid is not None and match.label_periodic_valid.shape != shape:
            raise ValueError('periodic validity shape must match input image')
    if evaluation is not None:
        if evaluation.ground_truth.shape != shape or evaluation.ground_truth_valid.shape != shape:
            raise ValueError('ground truth shapes must match input image')

    metadata: Diagnostics = dict(match.registration)
    metadata.update(format_version=1, image_shape=list(shape), labels_exported=bool(has_prediction),
                    demo_labels_exported=False,
                    prediction_status='available' if has_prediction else
                    'match_rejected' if not match.accepted else 'no_template_annotations',
                    annotation_status='exported' if has_prediction else
                    'blocked_by_match' if not match.accepted else 'not_available')
    artifacts = {'input': 'input.png', 'geometric_valid': 'geometric_valid.png'}
    if match.matrix_template_to_target is not None:
        artifacts['transform'] = 'matrix_template_to_target.npy'
    if has_prediction:
        artifacts.update(prediction_mask='mapped_labels.png', prediction_valid='label_valid.png')
        known = match.label_valid > 0
        ids, counts = np.unique(match.mapped_labels[known], return_counts=True)
        metadata.update(class_ids=[int(value) for value in ids],
                        class_pixels={str(int(value)): int(count) for value, count in zip(ids, counts)},
                        label_coverage_full_image=float(known.mean()))
        if match.label_periodic_valid is not None:
            artifacts['periodic_valid'] = 'label_periodic_valid.png'
    if evaluation is not None:
        metadata['evaluation'] = evaluation.metrics
        artifacts.update(ground_truth='ground_truth.png', ground_truth_valid='ground_truth_valid.png')
    metadata['artifacts'] = artifacts
    # Serialize before replacing files so invalid metadata cannot erase a good run.
    document = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        (out / name).unlink(missing_ok=True)
    write_image(out / artifacts['input'], match.input_image)
    write_image(out / artifacts['geometric_valid'], match.geometric_valid.astype(np.uint8) * 255)
    if 'transform' in artifacts:
        np.save(out / artifacts['transform'], match.matrix_template_to_target)
    if has_prediction:
        write_image(out / artifacts['prediction_mask'], match.mapped_labels)
        write_image(out / artifacts['prediction_valid'], known.astype(np.uint8) * 255)
        if 'periodic_valid' in artifacts:
            periodic = (match.label_periodic_valid > 0) & known
            write_image(out / artifacts['periodic_valid'], periodic.astype(np.uint8) * 255)
    if evaluation is not None:
        write_image(out / artifacts['ground_truth'], evaluation.ground_truth)
        write_image(out / artifacts['ground_truth_valid'], evaluation.ground_truth_valid.astype(np.uint8) * 255)
    manifest = out / 'result.json'
    manifest.write_text(document, encoding='utf-8')
    return manifest
