"""Stage 1: persist lossless prediction data for the independent viewer app."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import time

import numpy as np

from ...utils import validate_image, write_image
from ...utils.types import Diagnostics, PathInput
from ..models import EvaluationResult, MatchResult

SOURCE_LAYERS = ('direct', 'completion', 'object')

IMAGE_ARTIFACTS = (
    'input.png', 'geometric_valid.png', 'mapped_labels.png', 'label_valid.png',
    'label_periodic_valid.png', 'object_identities.png', 'object_valid.png',
    'base_labels.png', 'base_valid.png', 'ground_truth.png', 'ground_truth_valid.png',
    'direct_labels.png', 'direct_valid.png', 'completion_labels.png', 'completion_valid.png',
    'object_labels.png', 'object_layer_valid.png', 'object_layer_identities.png',
)

# Only these owned files are replaced on reruns, including obsolete reports.
ARTIFACTS = (
    'result.json', *IMAGE_ARTIFACTS, 'matrix_template_to_target.npy',
    'report.html', 'overview.jpg', 'best_alignment.jpg', 'label_overlay.jpg',
    'demo_source_labels.png', 'demo_mapped_labels.png', 'demo_label_valid.png',
    'demo_label_overlay.jpg', 'periodic_overlay.html', 'completed_alignment.jpg',
    'completed_geometric_valid.png', 'periodic_valid.png', 'completion_regions.jpg',
)

@dataclass(frozen=True)
class ExportPlan:
    match: MatchResult
    evaluation: EvaluationResult | None
    artifacts: dict[str, str]


def _plan_export(result: MatchResult | EvaluationResult) -> ExportPlan:
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
        if match.object_identities is not None:
            if (match.object_identities.shape != shape or match.object_identities.dtype != np.uint16
                    or match.object_valid is None or match.object_valid.shape != shape):
                raise ValueError('Object identity and validity arrays must match original input')
        if match.mapped_labels.shape != shape or match.label_valid.shape != shape:
            raise ValueError('prediction and validity shapes must match input image')
        if match.mapped_labels.dtype != np.uint16:
            raise ValueError('mapped_labels must be a uint16 class-ID map')
        if match.label_periodic_valid is not None and match.label_periodic_valid.shape != shape:
            raise ValueError('periodic validity shape must match input image')
    if evaluation is not None:
        if evaluation.ground_truth.shape != shape or evaluation.ground_truth_valid.shape != shape:
            raise ValueError('ground truth shapes must match input image')

    artifacts = {'input': 'input.png', 'geometric_valid': 'geometric_valid.png'}
    if match.matrix_template_to_target is not None:
        artifacts['transform'] = 'matrix_template_to_target.npy'
    if has_prediction:
        artifacts.update(prediction_mask='mapped_labels.png', prediction_valid='label_valid.png')
        for layer in SOURCE_LAYERS:
            label_key = layer+'_labels'
            valid_key = 'object_layer_valid' if layer == 'object' else layer+'_valid'
            layer_labels, layer_valid = getattr(match, label_key), getattr(match, valid_key)
            if layer_labels is not None or layer_valid is not None:
                if (layer_labels is None or layer_valid is None or layer_labels.shape != shape
                        or layer_labels.dtype != np.uint16 or layer_valid.shape != shape
                        or layer_valid.dtype != np.bool_):
                    raise ValueError(f'{layer} layer labels/validity must match original input and types')
                artifacts[label_key] = label_key+'.png'
                artifacts[valid_key] = valid_key+'.png'
        if match.object_layer_identities is not None:
            if ('object_labels' not in artifacts or match.object_layer_identities.shape != shape
                    or match.object_layer_identities.dtype != np.uint16):
                raise ValueError('Object layer identities require matching uint16 object labels')
            artifacts['object_layer_identities'] = 'object_layer_identities.png'
        if match.label_periodic_valid is not None:
            artifacts['periodic_valid'] = 'label_periodic_valid.png'
        if match.object_identities is not None:
            artifacts.update(object_identities='object_identities.png', object_valid='object_valid.png')
        if match.base_labels is not None:
            if match.base_labels.shape != shape or match.base_labels.dtype != np.uint16 or match.base_valid is None or match.base_valid.shape != shape:
                raise ValueError('Base labels and validity must match input image')
            artifacts.update(base_labels='base_labels.png', base_valid='base_valid.png')
    if evaluation is not None:
        artifacts.update(ground_truth='ground_truth.png', ground_truth_valid='ground_truth_valid.png')
    return ExportPlan(match, evaluation, artifacts)


def _json_document(plan: ExportPlan, artifacts: Mapping[str, str]) -> str:
    match = plan.match
    has_prediction = 'prediction_mask' in plan.artifacts
    labels_exported = {'prediction_mask', 'prediction_valid'} <= artifacts.keys()
    metadata: Diagnostics = dict(match.registration)
    metadata.update(format_version=1, image_shape=list(match.input_image.shape[:2]),
                    labels_exported=labels_exported, demo_labels_exported=False,
                    prediction_status='available' if has_prediction else
                    'match_rejected' if not match.accepted else 'no_template_annotations',
                    annotation_status='exported' if labels_exported else 'available' if has_prediction else
                    'blocked_by_match' if not match.accepted else 'not_available')
    if has_prediction:
        known = match.label_valid > 0
        ids, counts = np.unique(match.mapped_labels[known], return_counts=True)
        metadata.update(class_ids=[int(value) for value in ids],
                        class_pixels={str(int(value)): int(count) for value, count in zip(ids, counts)},
                        label_coverage_full_image=float(known.mean()))
        metadata['mask_layers'] = dict(overlap_allowed=True, final_prediction='exclusive',
            available=[layer for layer in SOURCE_LAYERS if layer+'_labels' in artifacts])
    if plan.evaluation is not None:
        metadata['evaluation'] = plan.evaluation.metrics
    metadata['artifacts'] = dict(artifacts)
    return json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)


def _write_images(plan: ExportPlan, out: Path, timings: dict[str, float] | None = None) -> None:
    match, evaluation, artifacts = plan.match, plan.evaluation, plan.artifacts

    def write(key, image):
        path = out / artifacts[key]
        if timings is None:
            write_image(path, image)
            return
        started = time.perf_counter()
        try:
            write_image(path, image)
        finally:
            timings['image_save_seconds'] += time.perf_counter() - started

    write('input', match.input_image)
    write('geometric_valid', match.geometric_valid.astype(np.uint8) * 255)
    if 'prediction_mask' in artifacts:
        known = match.label_valid > 0
        for layer in SOURCE_LAYERS:
            label_key = layer+'_labels'
            valid_key = 'object_layer_valid' if layer == 'object' else layer+'_valid'
            if label_key in artifacts:
                write(label_key, getattr(match, label_key))
                write(valid_key, getattr(match, valid_key).astype(np.uint8)*255)
        if 'object_layer_identities' in artifacts:
            write('object_layer_identities', match.object_layer_identities)
        if match.object_identities is not None:
            write('object_identities', match.object_identities)
            write('object_valid', match.object_valid.astype(np.uint8)*255)
        if match.base_labels is not None:
            write('base_labels', match.base_labels)
            write('base_valid', match.base_valid.astype(np.uint8)*255)
        write('prediction_mask', match.mapped_labels)
        write('prediction_valid', known.astype(np.uint8) * 255)
        if 'periodic_valid' in artifacts:
            periodic = (match.label_periodic_valid > 0) & known
            write('periodic_valid', periodic.astype(np.uint8) * 255)
    if evaluation is not None:
        write('ground_truth', evaluation.ground_truth)
        write('ground_truth_valid', evaluation.ground_truth_valid.astype(np.uint8) * 255)


def _write_json(out: Path, document: str) -> Path:
    manifest = out / 'result.json'
    staged = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=out, prefix='.result-', delete=False) as file:
            staged = Path(file.name)
            file.write(document)
        staged.replace(manifest)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return manifest


def save_images(result: MatchResult | EvaluationResult, output_dir: PathInput) -> dict[str, str]:
    """Write the result PNGs and return their manifest entries."""
    plan = _plan_export(result)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in IMAGE_ARTIFACTS:
        (out / name).unlink(missing_ok=True)
    _write_images(plan, out)
    return {key: name for key, name in plan.artifacts.items() if key != 'transform'}


def save_json(result: MatchResult | EvaluationResult, output_dir: PathInput, *,
              artifacts: Mapping[str, str] | None = None) -> Path:
    """Write metadata only. Pass the entries returned by save_images to link PNGs."""
    plan = _plan_export(result)
    links = {} if artifacts is None else dict(artifacts)
    out = Path(output_dir).resolve()
    required = {'input'}
    if 'prediction_mask' in plan.artifacts:
        required.update({'prediction_mask', 'prediction_valid'})
    if links and not required <= links.keys():
        raise ValueError(f'Missing result artifacts: {sorted(required - links.keys())}')
    for key, name in links.items():
        if key not in plan.artifacts or plan.artifacts[key] != name:
            raise ValueError(f'Invalid result artifact: {key}')
        path = (out / name).resolve()
        if not path.is_relative_to(out) or not path.is_file():
            raise ValueError(f'Invalid or missing result artifact: {key}')
    document = _json_document(plan, links)
    out.mkdir(parents=True, exist_ok=True)
    return _write_json(out, document)


def save_result(result: MatchResult | EvaluationResult, output_dir: PathInput, *,
                timings: dict[str, float] | None = None) -> Path:
    """Save PNGs, the transform and result.json for the viewer."""
    if timings is not None:
        timings['image_save_seconds'] = 0.
    plan = _plan_export(result)
    # Serialize before replacing files so invalid metadata cannot erase a good run.
    document = _json_document(plan, plan.artifacts)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        (out / name).unlink(missing_ok=True)
    if 'transform' in plan.artifacts:
        np.save(out / plan.artifacts['transform'], plan.match.matrix_template_to_target)
    _write_images(plan, out, timings)
    return _write_json(out, document)
